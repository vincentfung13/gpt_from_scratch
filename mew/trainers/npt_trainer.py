import os

import hydra
from omegaconf import DictConfig

from mew.nn.lm import TransformerLM
from mew.nn.functionals import cross_entropy_loss
from mew.optimizers.adamw import AdamW
from mew.optimizers.lr_scheduling import CosineAnnealingScheduler
from mew.data_loaders.numpy_batch_loader import NumpyBatchLoader
from mew.trainers.utils import save_checkpoint, load_checkpoint
from mew import LOGGER


class NPTTrainer:
    def __init__(self, cfg: DictConfig):
        # Init dataloader
        LOGGER.info(
            "[NTP_TRAINER] Init dataloader with batch_size=%d, seq_len=%d",
            cfg.batch_size,
            cfg.seq_len,
        )
        self.data_loader = NumpyBatchLoader(
            data=cfg.data,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
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
            rope_theta=cfg.model.rope.theta,
        )

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
            cosine_cycle_iters=cfg.optim.cosine_cycle_iters,
        )

        # Load checkpoint
        if cfg.trainer.resume:
            LOGGER.info(
                "[NPT_TRAINER] Resume training from checkpoint: %s",
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
                batch_size=self.cfg.batch_size,
                device=self.cfg.device,
            )

            # Forward
            logits = self.model(data)
            loss = cross_entropy_loss(logits, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # log progress
            if step > 0 and step % self.cfg.trainer.log_freq == 0:
                LOGGER.info(
                    "[NPT_TRAINER] Step [%d/%d], Loss: %.4f, LR: %.6f",
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
                LOGGER.info(f"[NPT_TRAINER] Checkpoint saved to {ckpt_path}")


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def launch_training(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    launch_training()
