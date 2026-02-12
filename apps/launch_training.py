from mew.trainers.npt_trainer import NPTTrainer


def _npt_training(cfg: DictConfig) -> None:
    trainer = NPTTrainer(cfg)
    trainer.train()


@hydra.main(version_base=None, config_path="cfgs", config_name="training")
def main(cfg: DictConfig) -> None:
    if cfg.task_name == "npt_training":
        _npt_training(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()