"""Offline student-top-k distillation on cached teacher trajectories."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from trl.experimental.gkd import GKDTrainer
from trl.models.utils import _ForwardRedirection
from trl.trainer.sft_trainer import SFTTrainer

from .direct_opd import topk_policy_loss


def topk_distillation_loss(student_logits, teacher_logits, top_k):
    """Direct-OPD's surrogate with one teacher: reward = log p_teacher - log p_student.

    Select the student's candidates and renormalize their weights within top-k.
    Both models' log probabilities retain full-vocabulary normalization.
    This scalar is a policy-gradient surrogate, not a nonnegative KL metric.
    """
    values, indices = student_logits.float().topk(min(top_k, student_logits.shape[-1]), dim=-1)
    student_logprobs = values - student_logits.float().logsumexp(dim=-1, keepdim=True)
    with torch.no_grad():
        teacher_logits = teacher_logits.float()
        teacher_logprobs = teacher_logits.gather(-1, indices) - teacher_logits.logsumexp(dim=-1, keepdim=True)
    return topk_policy_loss(student_logprobs, values, teacher_logprobs - student_logprobs.detach())[0]


class ChunkedTopKLoss(torch.nn.Module):
    """Project response positions in checkpointed chunks; chunk_size=-1 computes all at once."""

    def __init__(self, top_k=16, chunk_size=128):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if chunk_size != -1 and chunk_size < 1:
            raise ValueError("chunk_size must be -1 or positive")
        self.top_k = top_k
        self.chunk_size = chunk_size

    def _chunk_loss(self, student, student_weight, teacher, teacher_weight, student_bias, teacher_bias):
        student_logits = F.linear(student, student_weight, student_bias).float()
        with torch.no_grad():
            teacher_logits = F.linear(teacher, teacher_weight, teacher_bias).float()
        return topk_distillation_loss(student_logits, teacher_logits, self.top_k) * student.shape[0]

    def forward(self, student_input, student_weight, teacher_input, teacher_weight, true_labels,
                student_bias=None, teacher_bias=None):
        if student_weight.shape[0] != teacher_weight.shape[0]:
            raise ValueError("Student and teacher output vocabularies must have the same size")
        mask = true_labels.reshape(-1) != -100
        student = student_input.reshape(-1, student_input.shape[-1])[mask]
        teacher = teacher_input.detach().reshape(-1, teacher_input.shape[-1])[mask]
        count = student.shape[0]
        if not count:
            loss = student.sum() * 0 + student_weight.sum() * 0
            return loss if student_bias is None else loss + student_bias.sum() * 0
        if self.chunk_size == -1:
            return self._chunk_loss(student, student_weight, teacher, teacher_weight, student_bias, teacher_bias) / count
        total = student.new_zeros((), dtype=torch.float32)
        for start in range(0, count, self.chunk_size):
            args = (student[start:start + self.chunk_size], student_weight,
                    teacher[start:start + self.chunk_size], teacher_weight, student_bias, teacher_bias)
            if torch.is_grad_enabled() and (student.requires_grad or student_weight.requires_grad):
                value = checkpoint(self._chunk_loss, *args, use_reentrant=False)
            else:
                value = self._chunk_loss(*args)
            total = total + value
        return total / count


class OfflineGKDTrainer(GKDTrainer):
    def __init__(self, *args, loss_top_k=16, logit_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        # Reuse TRL's decoder-only forward and DDP redirection. Replace the fused
        # full-vocabulary JSD kernel with our native, checkpointed top-k head loss.
        self.use_liger_gkd_loss = True
        self._forward_redirection = _ForwardRedirection()
        self.liger_loss = ChunkedTopKLoss(loss_top_k, logit_chunk_size)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Cached teacher sequences stay fixed; never enter GKD's rollout branch.
        return SFTTrainer.training_step(self, model, inputs, num_items_in_batch)
