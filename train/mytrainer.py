"""Direct-OPD on student-generated trajectories, with vLLM weight synchronization."""

from pathlib import Path

import torch.nn.functional as F
from transformers import TrainerCallback
from trl import SFTTrainer
from trl.trainer.utils import disable_dropout_in_model

from .direct_opd import DirectOPDState, direct_opd_loss, tokenizer_source
from .student_generation import StudentGeneration
from .teacher_client import TeacherClient
from .utils import myDataCollator


class DirectOPDCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        opd = self.trainer.opd_state
        reward = opd.reward_sum / max(opd.reward_count, 1)
        # Accumulate all microbatches; use the resulting coefficient on the next optimizer step.
        opd.finish_step(args.adaptive_kl, args.alpha_epsilon, args.alpha_min, args.alpha_max)
        if state.global_step % args.logging_steps == 0:
            self.trainer.log({"opd/alpha": opd.alpha, "opd/dense_reward": reward, **self.trainer.opd_metrics})


class MyTrainer(SFTTrainer):
    def __init__(self, model, args, train_dataset, processing_class, opd_state=None, **kwargs):
        args.remove_unused_columns = False
        args.dataset_kwargs = {**(args.dataset_kwargs or {}), "skip_prepare_dataset": True}
        collator = kwargs.pop("data_collator", None)
        if collator is None:
            collator = myDataCollator(processing_class, args.max_length, args.enable_thinking)
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         processing_class=processing_class, data_collator=collator, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("Direct-OPD currently uses one training process and an external vLLM server")
        # Trainer scales gradient accumulation; this loss is a mean over answer tokens.
        self.model_accepts_loss_kwargs = False
        self.opd_state = opd_state or DirectOPDState(args.kl_beta, {})
        self.opd_metrics = {}
        self.add_callback(DirectOPDCallback(self))
        self.uses_reference = args.kl_beta > 0 or args.adaptive_kl
        if args.disable_dropout:
            disable_dropout_in_model(self.model)
        self.teacher_client = TeacherClient(
            model_before_path=args.model_before, model_after_path=args.model_after,
            student_tokenizer_path=tokenizer_source(args.student_model_path),
            reference_model_path=args.student_model_path if self.uses_reference else None,
            gpu=args.teacher_gpu, max_length=args.teacher_max_model_len,
            chunk_size=args.teacher_score_chunk_size,
        )
        try:
            self.student_generation = StudentGeneration(self.model, self.accelerator, self.processing_class, args,
                                                        state=self.opd_state)
        except BaseException:
            self.teacher_client.close()
            raise

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        answer_mask = inputs["labels"][:, 1:] != -100
        if not answer_mask.any():
            raise ValueError("Student rollout contains no answer tokens")
        logits = outputs.logits[:, :-1, :][answer_mask].float()
        topk_logits, indices = logits.topk(min(self.args.loss_top_k, logits.shape[-1]), dim=-1)
        lengths = answer_mask.sum(dim=1).tolist()
        candidates = [row.tolist() for row in indices.detach().cpu().split(lengths)]
        sampled_ids = inputs["labels"][:, 1:][answer_mask]
        sampled_candidates = [row.tolist() for row in sampled_ids[:, None].detach().cpu().split(lengths)]
        token_ids = [row[mask.bool()].tolist() for row, mask in zip(inputs["input_ids"], inputs["attention_mask"])]
        prompt_lengths = [len(row) - length for row, length in zip(token_ids, lengths)]
        # Both teachers see exactly the student's prompt and sampled continuation.
        before = self.teacher_client.score_before(token_ids, prompt_lengths, candidates).to(logits.device)
        after = self.teacher_client.score_after(token_ids, prompt_lengths, candidates).to(logits.device)
        logprobs = F.log_softmax(logits, dim=-1)
        sampled = logprobs.gather(-1, sampled_ids[:, None]).squeeze(-1)
        if self.uses_reference:
            reference = self.teacher_client.score_reference(token_ids, prompt_lengths, sampled_candidates)
            reference = reference.to(logits.device).squeeze(-1)
        else:
            reference = sampled.detach()
        loss, stats = direct_opd_loss(logprobs.gather(-1, indices), topk_logits, after, before,
                                      sampled, reference, self.opd_state.alpha)
        if model.training:
            self.opd_state.observe(stats["dense_reward"], indices.numel())
        self.opd_metrics = {"opd/" + name: float(value) for name, value in stats.items() if name != "dense_reward"}
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = dict(inputs)
        ids, mask, labels, completions = self.student_generation.generate(inputs)
        inputs.update(input_ids=ids, attention_mask=mask, labels=labels, completion_ids=completions)
        return super().training_step(model, inputs, num_items_in_batch)

    def train(self, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint is not None:
            self.opd_state.restore(resume_from_checkpoint)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            self.opd_state.save(Path(output_dir or self.args.output_dir), self.state.global_step)

    def close(self):
        try:
            self.student_generation.close()
        finally:
            self.teacher_client.close()
