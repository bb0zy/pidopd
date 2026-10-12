
import random
import textwrap
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from trl import SFTTrainer
from trl.experimental.utils import empty_cache
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import disable_dropout_in_model

from .myconfig import MyConfig
from .utils import myDataCollator
from .vllm_client import VLLMClient


if is_peft_available():
    from peft import PeftConfig


logger = get_logger(__name__)


class MyTrainer(SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: str | None = None,
        args: MyConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
    ):
        # Ensure Trainer does not drop non-signature columns used by the collator (e.g., "prompts")
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = myDataCollator(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using a ChatML collator,
        # so that raw conversational fields (e.g., "messages") remain available to the collator.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        # The loss uses a per-micro-batch token mean; let Trainer scale gradient accumulation.
        self.model_accepts_loss_kwargs = False

        # teacher_model is the served API model name, not a local model path.
        self.teacher_client = VLLMClient(
            base_url=args.teacher_vllm_base_url,
            model=teacher_model or args.teacher_vllm_model,
            api_key=args.teacher_vllm_api_key,
        )

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.beta = args.beta
        self.temperature = args.temperature
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            # unwrap_model_for_generation temporarily disables gradient checkpointing.
            "use_cache": True,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
 
        self.generation_kwargs = generation_kwargs
    
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        num_items_in_batch=None,
    ):
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if num_items_in_batch is not None:
            # Normalize by the global number of valid tokens for gradient-accumulation-correct loss (see issue #4719).
            jsd_sum = jsd.sum()
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
            return jsd_sum / num_items_in_batch
        if reduction == "batchmean":
            # clamp_min(1) avoids 0/0 -> nan when a sample has no unmasked positions
            # (e.g. completion fully truncated). jsd[mask] is empty -> jsd.sum() == 0,
            # so 0/1 == 0 with a valid grad path.
            denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
            return jsd.sum() / denom
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Use the same teacher model with normal and privileged prompts.
        normal_teacher_inputs = []
        normal_teacher_prompt_lengths = []
        teacher_inputs = []
        teacher_prompt_lengths = []
        student_prompt_length = inputs["prompts"].size(1)
        for i in range(inputs["input_ids"].size(0)):
            input_mask = inputs["attention_mask"][i].bool()
            normal_teacher_inputs.append(inputs["input_ids"][i][input_mask].tolist())

            prompt_mask = inputs["teacher_prompt_attention_mask"][i].bool()
            prompt_ids = inputs["teacher_prompts"][i][prompt_mask].tolist()
            completion_mask = inputs["attention_mask"][i, student_prompt_length:].bool()
            completion_ids = inputs["completion_ids"][i][completion_mask].tolist()
            normal_teacher_prompt_lengths.append(len(normal_teacher_inputs[i]) - len(completion_ids))
            teacher_inputs.append(prompt_ids + completion_ids)
            teacher_prompt_lengths.append(len(prompt_ids))

        normal_teacher_response = self.teacher_client.score(normal_teacher_inputs)
        teacher_response = self.teacher_client.score(teacher_inputs)

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )

        # Student logits at position t - 1 predict the answer token at position t.
        answer_mask = inputs["labels"][:, 1:] != -100
        student_logits = student_outputs.logits[:, :-1, :][answer_mask].float()
        topk_logits, topk_indices = student_logits.topk(self.args.loss_top_k, dim=-1)
        topk_indices_cpu = topk_indices.detach().cpu()

        # score() omits the first input token, so answer scores start at prompt_length - 1.
        # Select matching student candidates on CPU before transferring rewards to the GPU.
        teacher_topk_diffs = []
        offset = 0
        for i, prompt_length in enumerate(teacher_prompt_lengths):
            teacher_log_probs = teacher_response[i][prompt_length - 1:]
            normal_prompt_length = normal_teacher_prompt_lengths[i]
            normal_teacher_log_probs = normal_teacher_response[i][normal_prompt_length - 1:]
            answer_length = teacher_log_probs.size(0)
            indices = topk_indices_cpu[offset : offset + answer_length]
            teacher_topk = teacher_log_probs.gather(dim=-1, index=indices)
            normal_teacher_topk = normal_teacher_log_probs.gather(dim=-1, index=indices)
            teacher_topk_diffs.append(teacher_topk - normal_teacher_topk)
            offset += answer_length

        # Only the selected log probability gains, shaped (answer_tokens, K), move to the GPU.
        r_1 = torch.cat(teacher_topk_diffs, dim=0).detach().to(student_logits.device)
        topk_weights = F.softmax(topk_logits.detach(), dim=-1)
        r2 = topk_weights * r_1

        # Normalize over the full vocabulary and keep gradients only through student log probabilities.
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        topk_log_probs = student_log_probs.gather(dim=-1, index=topk_indices)
        loss = -(r2 * topk_log_probs).sum(dim=-1).mean()

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config):
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs["prompt_attention_mask"],
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        device = generated_tokens.device
        prompt_mask = inputs["prompt_attention_mask"]
        prompt_length = inputs["prompts"].shape[1]
        completion_ids = generated_tokens[:, prompt_length:]

        # Mask everything after the first EOS token. eos_token_id can be a list of stop tokens (Llama 3),
        # so match with torch.isin; with no eos id nothing stops early, so keep the whole completion.
        if generation_config.eos_token_id is None:
            completion_mask = torch.ones_like(completion_ids)
        else:
            is_eos = torch.isin(completion_ids, torch.tensor(generation_config.eos_token_id, device=device))
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Pad ids can appear inside real prompt text, so use the prompt mask instead of matching ids
        new_attention_mask = torch.ones_like(generated_tokens)
        new_attention_mask[:, prompt_length:] = completion_mask
        new_attention_mask[:, :prompt_length] = prompt_mask

        new_labels = generated_tokens.clone()
        new_labels[new_attention_mask == 0] = -100

        # Mask the prompt so only the generated completion contributes to the loss. `generate` echoes
        # the prompt back as the first `prompt_length` columns, so masking them with -100 matches the
        # collator convention (`labels[:len(prompt)] = -100`) that `compute_loss` relies on.
        new_labels[:, :prompt_length] = -100

        return generated_tokens, new_attention_mask, new_labels, completion_ids

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
     
      
        with (
            unwrap_model_for_generation(
                model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
            ) as unwrapped_model
        ):
            new_input_ids, new_attention_mask, new_labels, completion_ids = self.generate_on_policy_outputs(
                unwrapped_model, inputs, self.generation_config
            )
        inputs["input_ids"] = new_input_ids
        inputs["attention_mask"] = new_attention_mask
        inputs["labels"] = new_labels
        inputs["completion_ids"] = completion_ids

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
