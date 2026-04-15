# mew 

<img src="resources/logo.jpg" width="59%" alt="mew logo">

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
uv run apps/tokenization.py
```

**Training:**
```bash
uv run apps/launch_training.py
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
