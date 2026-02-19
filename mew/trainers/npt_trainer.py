import os
import logging
from tqdm import tqdm

from omegaconf import DictConfig

import torch

from mew.nn.utils import build_model
from mew.nn.functionals import cross_entropy
from mew.optimizers.adamw import AdamW
from mew.optimizers.lr_scheduling import CosineAnnealingScheduler
from mew.optimizers.utils import clip_gradients
from mew.data_loaders.numpy_batch_loader import NumpyBatchLoader
from mew.trainers.utils import save_checkpoint, load_checkpoint

LOGGER = logging.getLogger(__name__)


torch.autograd.set_detect_anomaly(True)


class NPTTrainer:
    def __init__(self, cfg: DictConfig, wandb=None):
        # Init dataloader
        LOGGER.info(
            "Init train dataloader from %s with seq_len=%d",
            cfg.data.train_file,
            cfg.data.seq_len,
        )
        self.train_data_loader = NumpyBatchLoader(
            data_path=cfg.data.train_file,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
        )
        LOGGER.info(
            "Starting running evalidation, init val dataloader from %s with seq_len=%d",
            cfg.data.val_file,
            cfg.data.seq_len,
        )
        self.val_data_loader = NumpyBatchLoader(
            data_path=cfg.data.val_file,
            seq_len=cfg.data.seq_len,
            batch_size=cfg.data.batch_size,
        )

        # Init model
        LOGGER.info("Init LM model and optimizer")
        self.model = build_model(
            cfg=cfg,
            device=cfg.device,
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

        self.wandb = wandb

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
        for step in range(1, self.cfg.trainer.total_steps + 1):
            # Get batch
            data, target = self.train_data_loader.get_batch(
                device=self.cfg.device,
            )

            # Forward
            logits = self.model(data)
            loss = cross_entropy(logits, target)
            scaled_loss = loss / self.cfg.optim.grad_accumulation_steps
            scaled_loss.backward()

            # Backward
            if step % self.cfg.optim.grad_accumulation_steps == 0:
                if self.cfg.optim.grad_clip_norm > 0:
                    clip_gradients(
                        parameters=self.model.parameters(),
                        max_l2_norm=self.cfg.optim.grad_clip_norm,
                    )
                self.optim.step()
                self.lr_scheduler.step()
                self.optim.zero_grad()

            # log progress
            if step % self.cfg.trainer.log_freq == 0:
                # Run mini val
                with torch.no_grad():
                    val_data, val_target = self.val_data_loader.get_batch(
                        device=self.cfg.device
                    )
                    val_logits = self.model(val_data)
                    val_loss = cross_entropy(val_logits, val_target)

                LOGGER.info(
                    "Step [%d/%d], Train Loss: %.4f, Val Loss: %.4f LR: %.6f",
                    step,
                    self.cfg.trainer.total_steps,
                    loss.item(),
                    val_loss.item(),
                    self.optim.param_groups[0]["lr"],
                )
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "train/loss": loss.item(),
                            "val/loss": val_loss.item(),
                            "train/lr": self.optim.param_groups[0]["lr"],
                        },
                        step=step,
                    )

            # Save checkpoint
            if step % self.cfg.trainer.save_freq == 0:
                if not os.path.isdir(self.cfg.save_dir):
                    os.makedirs(self.cfg.save_dir)
                ckpt_path = os.path.join(
                    self.cfg.save_dir,
                    f"checkpoint_step_{step}.pt",
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optim,
                    iteration=step,
                    output_path=ckpt_path,
                    lr_scheduler=self.lr_scheduler,
                )
                LOGGER.info(f"Checkpoint saved to {ckpt_path}")

    def _run_val(self, step: int) -> float:
        LOGGER.info(
            "Starting running evalidation, init val dataloader from %s with seq_len=%d",
            self.cfg.data.val_file,
            self.cfg.data.seq_len,
        )
        val_data_loader = NumpyBatchLoader(
            data_path=self.cfg.data.val_file,
            seq_len=self.cfg.data.seq_len,
            batch_size=self.cfg.data.batch_size,
        )

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            total_losses = 0.0
            sample_count = 0
            for _ in tqdm(list(range(self.cfg.trainer.val_steps))):
                data, target = val_data_loader.get_batch(
                    device=self.cfg.device,
                )
                logits = self.model(data)
                loss = cross_entropy(logits, target)
                total_losses += loss * data.size()[0]
                sample_count += data.size()[0]
            avg_val_loss = (total_losses / sample_count).item()
            LOGGER.info("Validation Loss: %.4f", avg_val_loss)
            if self.wandb is not None:
                self.wandb.log(
                    {
                        "val/loss": avg_val_loss,
                    },
                    step=step,
                )
        self.model.train()
        return avg_val_loss
