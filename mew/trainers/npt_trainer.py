import os
import logging

import hydra
from omegaconf import DictConfig

import torch

from mew.nn.lm import TransformerLM
from mew.nn.functionals import cross_entropy
from mew.optimizers.adamw import AdamW
from mew.optimizers.lr_scheduling import CosineAnnealingScheduler
from mew.data_loaders.numpy_batch_loader import NumpyBatchLoader
from mew.trainers.utils import save_checkpoint, load_checkpoint

LOGGER = logging.getLogger(__name__)


torch.autograd.set_detect_anomaly(True)


class NPTTrainer:
    def __init__(self, cfg: DictConfig):
        # Init dataloader
        LOGGER.info(
            "[NTP_TRAINER] Init dataloader from %s with seq_len=%d",
            cfg.data.train_file,
            cfg.data.seq_len,
        )
        self.data_loader = NumpyBatchLoader(
            data_path=cfg.data.train_file,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
        )

        # Init model
        LOGGER.info("[NTP_TRAINER] Init LM model and optimizer")
        self.model = TransformerLM(
            d_model=cfg.model.d_model,
            d_ff=cfg.model.d_ff,
            num_heads=cfg.model.num_heads,
            vocab_size=cfg.model.vocab_size,
            context_len=cfg.model.context_len,
            num_transformer_layers=cfg.model.num_transformer_layers,
            rope_theta=cfg.model.rope_theta,
        ).cuda()

        # Init optimizer and lr scheduler
        self.optim = AdamW(
            params=self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
            eps=cfg.optim.eps,
        )
        self.lr_scheduler = CosineAnnealingScheduler(
            optimizer=self.optim,
            max_learning_rate=cfg.optim.lr,
            min_learning_rate=cfg.optim.min_lr,
            warmup_iters=cfg.optim.warmup_iters,
            cosine_cycle_iters=cfg.optim.cosine_cyle_iters,
        )

        # Load checkpoint
        if cfg.trainer.resume:
            LOGGER.info(
                "Resume training from checkpoint: %s",
                cfg.trainer.resume_checkpoint_path,
            )
            load_checkpoint(
                src=cfg.trainer.resume_checkpoint_path,
                model=self.model,
                optimizer=self.optim,
                lr_scheduler=self.lr_scheduler,
            )

        self.cfg = cfg

    def train(self):
        # Main training loop
        for step in range(self.cfg.trainer.total_steps):
            # Get batch
            data, target = self.data_loader.get_batch(
                device=self.cfg.device,
            )

            # Forward
            logits = self.model(data)
            loss = cross_entropy(logits, target)

            # Backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.lr_scheduler.step()

            # log progress
            if step > 0 and step % self.cfg.trainer.log_freq == 0:
                LOGGER.info(
                    "Step [%d/%d], Loss: %.4f, LR: %.6f",
                    step,
                    self.cfg.trainer.total_steps,
                    loss.item(),
                    self.optim.param_groups[0]["lr"],
                )

            # Save checkpoint
            if step > 0 and step % self.cfg.trainer.save_freq == 0:
                ckpt_path = os.path.join(
                    self.cfg.trainer.save_dir,
                    f"checkpoint_step_{step}.pt",
                )
                save_checkpoint(
                    dst=ckpt_path,
                    model=self.model,
                    optimizer=self.optim,
                    lr_scheduler=self.lr_scheduler,
                )
                LOGGER.info(f"Checkpoint saved to {ckpt_path}")


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def launch_training(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    launch_training()
