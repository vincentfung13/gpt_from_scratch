# mew 

<p align="center">
    <img src="resources/logo.jpg" width="30%" alt="mew logo">
</p>

A custom implementation of a GPT-like language model developed entirely from scratch. This project provides the fundamental building blocks for training and running inference on an autoregressive neural probabilistic language model.

## Features

- **Custom Tokenization**: Byte-Pair Encoding (BPE) implementation.
- **Efficient Data Loading**: Memory-optimized Numpy batch loaders capable of handling massive memmap files.
- **Neural Network Architecture**: Custom Transformer blocks, Rotary Positional Embeddings (RoPE), and linear layers built with PyTorch.
- **Optimizers**: Custom AdamW optimizer with learning rate scheduling.
- **Generators**: Autoregressive text generation logic.
- **Trainer**: Training loops with experiment tracking (W&B integration).
- **Configuration Management**: Hydra-based configuration for easy parameter sweeping and experiment management.

## Project Structure

The repository is divided into two primary packages:

### 1. `@mew/` (Core Library)
The core engine behind the language model:
- `mew/data_loaders/`: Efficient batching and data loading logic (`numpy_batch_loader`).
- `mew/generators/`: Text generation utilities (`conditional_generator`).
- `mew/nn/`: Neural network architectures, modules, and layers (Transformers, RoPE).
- `mew/optimizers/`: Custom optimizers and schedulers (AdamW, LR scheduling).
- `mew/tokenization/`: BPE tokenizer and text processing tools.
- `mew/trainers/`: Implementations of the training loops (e.g., `NPTTrainer`).

### 2. `@apps/` (Application Layer)
High-level scripts and configurations:
- `apps/cfgs/`: Hydra configuration files (`training.yaml`, `inference.yaml`, `tokenization.yaml`).
- `apps/launch_training.py`: Entry point for launching model training.
- `apps/tokenization.py`: Entry point for running the data tokenization pipelines.

## Setup and Installation

This project strictly uses **`uv`** for fast and reliable Python package management. 

1. Ensure you have `uv` installed.
2. Install the project and its dependencies:
   ```bash
   uv sync
   ```

## Usage

You can run the application scripts using `uv run`. 

**Tokenization:**
```bash
# 1. Train a BPE tokenizer
uv run apps/tokenization.py \
    task_name=train_bpe \
    training.vocab_size=10000 \
    training.input_path='./data/TinyStoriesV2-GPT4-train.txt' \
    training.save_dir='./data/tinystories_bpe_tokenizer'

# 2. Tokenize the training and vaidation file
uv run apps/tokenization.py \
    task_name=tokenize_file \
    file_tokenization.input_path='./data/TinyStoriesV2-GPT4-train.txt' \
    file_tokenization.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    file_tokenization.save_path='./data/tiny_stories_train.tokens.uint16.npy' \
    file_tokenization.num_workers=12

uv run apps/tokenization.py \
    task_name=tokenize_file \
    file_tokenization.input_path='./data/TinyStoriesV2-GPT4-valid.txt' \
    file_tokenization.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    file_tokenization.save_path='./data/tiny_stories_val.tokens.uint16.npy' \
    file_tokenization.num_workers=12
```

**Training:**
```bash
EXP_PREFIX="tinystories_ablation_$(date +%Y%m%d_%H%M%S)"
uv run apps/launch_training.py \
    run_name="${EXP_PREFIX}_mha" \
    trainer.total_steps=10000 \
    data.train_file='./data/tiny_stories_train.tokens.uint16.npy' \
    data.val_file='./data/tiny_stories_val.tokens.uint16.npy' \
    data.batch_size=128 \
    data.seq_len=256 \
    data.tokenizer_path='./data/tinystories_bpe_tokenizer' \
    model.d_model=512 \
    model.d_ff=1344 \
    model.num_heads=16 \
    model.num_groups=null
```
*Note: The launch scripts use Hydra, so you can override configurations via the CLI (e.g., `uv run apps/launch_training.py wandb.enable=True`).*

## Development Guidelines

- **Formatting**: Always format the code using `black`.
- **Linting**: Check for lint errors using `flake8`, but ignore the "line too long" error (`E501`).

```bash
uv run black mew/ apps/
uv run flake8 --ignore=E501 mew/ apps/
```

See [AGENTS.md](AGENTS.md) for more details regarding instructions for AI agents and code contributors.
