"""Check the KL regularizer against known probability distributions."""

import importlib.util
import math
import unittest


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "trl", "transformers", "datasets", "accelerate")),
    "Requires the training framework; all checks run on CPU without model downloads",
)
class LossRegularizationTests(unittest.TestCase):
    def test_coarse_kl_includes_tail_and_backpropagates(self):
        import torch
        from train.mytrainer import coarse_kl_loss

        p = [0.40, 0.30, 0.15, 0.10, 0.05]
        q = [0.20, 0.15, 0.30, 0.25, 0.10]
        policy_logits = torch.tensor([list(map(math.log, p))], requires_grad=True)
        policy_log_probs = policy_logits.log_softmax(dim=-1)
        reference_log_probs = torch.tensor([list(map(math.log, q))])
        indices = torch.tensor([[0, 1, 2]])
        actual = coarse_kl_loss(
            policy_log_probs.gather(-1, indices), reference_log_probs.gather(-1, indices),
        )
        expected = sum(p[i] * math.log(p[i] / q[i]) for i in (0, 1, 2))
        expected += 0.15 * math.log(0.15 / 0.35)
        self.assertAlmostEqual(actual.item(), expected, places=6)
        self.assertLessEqual(actual.item(), sum(p_i * math.log(p_i / q_i) for p_i, q_i in zip(p, q)))
        actual.backward()
        self.assertTrue(torch.isfinite(policy_logits.grad).all())
        self.assertGreater(policy_logits.grad.abs().sum().item(), 0)

        same_reference = policy_log_probs.detach().gather(-1, indices)
        same_kl = coarse_kl_loss(policy_log_probs.gather(-1, indices), same_reference)
        self.assertAlmostEqual(same_kl.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
