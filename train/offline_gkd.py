"""GKD loss and checkpointing with generation completely disabled."""

from trl.experimental.gkd import GKDTrainer
from trl.trainer.sft_trainer import SFTTrainer


class OfflineGKDTrainer(GKDTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Keep GKD's compute_loss; bypass its stochastic rollout branch entirely.
        return SFTTrainer.training_step(self, model, inputs, num_items_in_batch)
