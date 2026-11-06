"""Direct-OPD loss checks against explicit distributions and gradient directions."""

import unittest

import torch

from train.direct_opd import direct_opd_loss, low_var_kl


class LossRegularizationTests(unittest.TestCase):
    def test_loss_matches_explicit_teacher_shift_plus_sampled_kl(self):
        logits = torch.tensor([[0.4, 1.0, -0.7], [0.8, -0.2, 0.1]], requires_grad=True)
        indices = torch.tensor([[1, 0], [0, 2]])
        before = torch.tensor([[-0.5, -1.5], [-1.0, -0.6]], requires_grad=True)
        after = torch.tensor([[-0.1, -1.7], [-0.3, -0.9]], requires_grad=True)
        logprobs = logits.log_softmax(-1)
        selected = logprobs.gather(-1, indices)
        topk = logits.gather(-1, indices)
        sampled = logprobs[:, 0]
        anchor = torch.tensor([-0.5, -0.4], requires_grad=True)
        loss, stats = direct_opd_loss(selected, topk, after, before, sampled, anchor, 0.7)
        expected_policy = -(topk.softmax(-1).detach() * (after - before).detach() * selected).sum(-1).mean()
        delta = anchor.detach() - sampled
        expected_kl = (delta.exp() - delta - 1).mean()
        torch.testing.assert_close(loss, expected_policy + 0.7 * expected_kl)
        torch.testing.assert_close(stats['dense_reward'], (topk.softmax(-1) * (after - before)).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIsNone(before.grad)
        self.assertIsNone(after.grad)
        self.assertIsNone(anchor.grad)

    def test_positive_teacher_shift_increases_probability_and_weights_are_detached(self):
        logprobs = torch.zeros(1, 2, requires_grad=True)
        logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
        after = torch.tensor([[2.0, -1.0]])
        sampled = torch.zeros(1)
        loss, _ = direct_opd_loss(logprobs, logits, after, torch.zeros_like(after), sampled, sampled, 0)
        loss.backward()
        self.assertLess(logprobs.grad[0, 0].item(), 0)
        self.assertGreater(logprobs.grad[0, 1].item(), 0)
        self.assertIsNone(logits.grad)

    def test_identical_teachers_produce_zero_policy_update(self):
        logprobs = torch.randn(2, 3, requires_grad=True)
        teacher = torch.randn(2, 3)
        sampled = torch.zeros(2)
        loss, _ = direct_opd_loss(logprobs, logprobs, teacher, teacher, sampled, sampled, 0)
        loss.backward()
        torch.testing.assert_close(logprobs.grad, torch.zeros_like(logprobs))

    def test_k3_is_zero_at_anchor_and_pulls_in_both_directions(self):
        reference = torch.zeros(2)
        self.assertTrue(torch.equal(low_var_kl(reference, reference), reference))
        moved = torch.tensor([1.0, -1.0], requires_grad=True)
        low_var_kl(moved, reference).sum().backward()
        self.assertGreater(moved.grad[0].item(), 0)
        self.assertLess(moved.grad[1].item(), 0)


if __name__ == '__main__':
    unittest.main()
