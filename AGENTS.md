# Agent Instructions

This repository contains a custom implementation of a GPT-like language model developed from scratch.

## 0. Purpose: This Is a Learning Project

This repo exists so the user can learn how a GPT-like model works by implementing it themselves, by hand. This is the most important instruction in this file and overrides convenience.

- **Do not write or edit the core implementation for the user.** This includes model/layer code (`mew/nn/`), tokenization logic (`mew/tokenization/`), optimizers (`mew/optimizers/`), data loaders (`mew/data_loaders/`), generation logic (`mew/generators/`), and training loops (`mew/trainers/`).
- **Instead, give suggestions, hints, and explanations.** Point to the relevant concept, paper, algorithm, or a similar pattern already in the codebase, and let the user write the code. Ask Socratic questions if the user seems stuck, rather than supplying the answer outright.
- **Debugging is the exception.** If the user's own code has a bug, you may read the code, help diagnose the root cause, and explain the fix. Prefer explaining the bug and letting the user apply the fix; only write the fix directly if the user asks you to or it's a trivial one-line correction to code they already wrote.
- **Non-core work is fine to implement directly**, e.g. Hydra configs, scripts under `apps/`, tooling, formatting/lint fixes, tests, documentation, or plumbing that isn't itself the learning exercise. When unsure whether something counts as "core," ask.

## 1. Background

The codebase provides the core building blocks to train and run inference on a neural probabilistic language model. It includes custom tokenization (BPE), data loading, neural network layers (Transformers, RoPE), optimizers (AdamW with learning rate scheduling), text generation, and training loops. The project allows users to understand and experiment with the fundamental components of modern generative AI models.

1. 2\. High-Level Design and Modules

The architecture is cleanly separated into two main packages:

- **`@mew/`** **(Core Library):**
  - `mew/data_loaders/`: Handles batching and loading data for training (e.g., `numpy_batch_loader`).
  - `mew/generators/`: Contains logic for autoregressive text generation (e.g., `conditional_generator`).
  - `mew/nn/`: Implements the neural network architecture, including Transformer blocks, linear layers, and rotary positional embeddings (RoPE).
  - `mew/optimizers/`: Provides optimization algorithms like AdamW and custom learning rate scheduling.
  - `mew/tokenization/`: Contains the custom Byte-Pair Encoding (BPE) tokenizer and text processing utilities.
  - `mew/trainers/`: Implements the training loops and utilities for training the language model (e.g., `NPTTrainer`).
- **`@apps/`** **(Application Layer):**
  - Contains high-level scripts to execute workflows using the `mew` library.
  - `apps/cfgs/`: Stores Hydra configurations for tokenization, training, and inference.
  - `apps/launch_training.py` & `apps/tokenization.py`: Entry points for launching model training and running the data tokenization pipelines.

## 3. Package Management

- **Always use** **`uv`** for package management and running the code.
- Example: Use `uv run <script.py>` to execute code or `uv pip install <package>` for managing dependencies to ensure a fast, reliable, and reproducible Python environment.

## 4. Code Formatting and Linting

- **Always format the code with** **`black`.**
- **Check for lint errors with** **`flake8`**, but strictly ignore the "line too long" error (`E501`).
- **Scope:** Only apply `uvx black` formatting and `uvx flake8` linting to the core packages `@mew/` and `@apps/`. Do not run them on other directories or files in the repository.

