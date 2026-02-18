import hydra
import os
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def main(cfg: DictConfig) -> None:
    # Copy tokenizer to save dir
    os.system(f"cp -r {cfg.data.tokenizer_path} {cfg.save_dir}/tokenizer")

    # Launch training job
    if cfg.task_name == "npt_training":
        from mew.trainers.npt_trainer import NPTTrainer

        trainer = NPTTrainer(cfg)
        trainer.train()
    else:
        raise ValueError(f"Unknown task_name: {cfg.task_name}")


if __name__ == "__main__":
    main()
