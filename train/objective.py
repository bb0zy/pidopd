"""Direct-OPD top-1 surrogate (paper Eqs. 10--16)."""
import torch


def selected_log_probs(logits, token_ids, chunk_size=64):
    """Full-vocabulary log probabilities without allocating a full FP32 softmax."""
    parts = []
    for start in range(0, logits.shape[0], chunk_size):
        block = logits[start:start + chunk_size].float()
        ids = token_ids[start:start + chunk_size]
        parts.append(block.gather(-1, ids[:, None]).squeeze(-1) - block.logsumexp(-1))
    return torch.cat(parts)


def loss_terms(top1_logp, sampled_logp, teacher_shift, reference_logp):
    # For the paper's renormalized singleton support p_bar=1, not p_full(top1).
    policy = -(teacher_shift.detach() * top1_logp).mean()
    # Sampled k3 KL surrogate, NOT an exact full-vocabulary KL gradient.
    delta = reference_logp.detach() - sampled_logp
    kl = (torch.expm1(delta) - delta).mean()
    return policy, kl


def update_alpha(alpha, mean_shift, rate=0.01, minimum=0.5, maximum=2.5):
    sign = (mean_shift > 0) - (mean_shift < 0)
    return min(maximum, max(minimum, alpha * (1 + rate * sign)))
