import hydra
from omegaconf import DictConfig

from mew.tokenization.bpe import BPETokenizer
from mew.tokenization.file_processor import FileProcessor


def _train_bpe(cfg: DictConfig) -> None:
    bpe = BPETokenizer()
    bpe.train(
        input_path=cfg.training.input_path,
        save_dir=cfg.training.save_dir,
        vocab_size=cfg.training.vocab_size,
        special_tokens=cfg.special_tokens,
        file_split_token=cfg.file_split_token,
        pre_tokenization_pattern=cfg.pre_tokenization_pattern,
    )


def _tokenize_file(cfg: DictConfig) -> None:
    special_tokens = cfg.special_tokens
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    fp = FileProcessor(file_path=cfg.file_tokenization.input_path)
    fp.tokenize_file(
        tokenizer_path=cfg.file_tokenization.tokenizer_path,
        pre_tokenization_pattern=cfg.pre_tokenization_pattern,
        special_tokens=special_tokens,
        output_path=cfg.file_tokenization.save_path,
        num_workers=cfg.file_tokenization.num_workers,
    )


@hydra.main(version_base=None, config_path="cfgs", config_name="tokenization")
def main(cfg: DictConfig) -> None:
    if cfg.task_name == "train_bpe":
        _train_bpe(cfg)
    elif cfg.task_name == "tokenize_file":
        _tokenize_file(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
