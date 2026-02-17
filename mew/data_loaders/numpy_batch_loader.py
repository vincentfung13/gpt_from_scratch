import torch
import numpy as np
import logging
from typing import Tuple

LOGGER = logging.getLogger(__name__)


class NumpyBatchLoader:
    def __init__(self, data_path: str, seq_len: int, batch_size: int, dtype=np.uint16):
        """
        Memory-optimized loader for massive memmap files.
        """
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_num = len(self.data)

        # Instead of every index, we track "chunks" or "offsets"
        # Total valid starting positions
        self.num_samples = self.token_num - seq_len - 1
        # We calculate how many batches we can fit in one epoch
        self.num_batches = self.num_samples // batch_size

        # We only store the batch count, not every index
        self.batch_indices = np.arange(self.num_batches)
        self.current_batch_idx = 0

        self._shuffle_indices()
        LOGGER.info(
            f"Mapped {self.token_num} tokens. "
            f"Approx {self.num_batches} batches per epoch."
        )

    def _shuffle_indices(self):
        """Shuffles the batch order at the start of an epoch."""
        np.random.shuffle(self.batch_indices)
        self.current_batch_idx = 0

    def get_batch(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current_batch_idx >= self.num_batches:
            LOGGER.info("Epoch complete. Re-shuffling...")
            self._shuffle_indices()

        # Get the starting offset for this batch
        # We jump by batch_size to ensure distinct windows
        start_idx = self.batch_indices[self.current_batch_idx] * self.batch_size
        self.current_batch_idx += 1

        # Generate indices for the current batch
        # Note: This samples contiguous blocks which is standard for LLM pretraining efficiency,
        # but the order of these blocks is randomized.
        indices = np.arange(start_idx, start_idx + self.batch_size)

        # Efficient batch construction
        x_list, y_list = [], []
        for i in indices:
            # Slicing memmap is O(1) memory as it returns a view
            x_list.append(self.data[i : i + self.seq_len])
            y_list.append(self.data[i + 1 : i + self.seq_len + 1])

        # Convert to torch directly from numpy views
        x = torch.from_numpy(np.stack(x_list)).to(device).long()
        y = torch.from_numpy(np.stack(y_list)).to(device).long()

        return x, y
