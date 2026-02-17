import torch
import os
import numpy as np
import logging
from typing import Union, Tuple

LOGGER = logging.getLogger(__name__)


class NumpyBatchLoader:
    def __init__(self, data: Union[str, np.memmap], seq_len: int, batch_size: int = -1):
        """
        Creates a data loader that loads the training data and prepares batches for training for 1 epoch.

        Arguments:
            data: either a path to a numpy memmap file or a numpy memmap object.
                data should be of size (total_number_of_token, )
            seq_len: training seq len
        """
        if isinstance(data, str):
            assert os.path.exists(data)
            self.data = np.memmap(filename=data, dtype=np.uint16, mode="r")
        else:
            self.data = data
        self.seq_len = seq_len
        self.token_num = len(data)
        self.available_indices = set(np.arange(self.token_num - self.seq_len).tolist())
        LOGGER.info(f"[DATA_LOADER] Loaded {self.token_num} tokens.")

    def get_batch(
        self,
        batch_size: int,
        device: str,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Arguments:
            batch_size: training batch size
            device: "cpu|gpu|mps (for apple silicon)"

        Returns: -> Tuple[data, target], where:
            data: a batch of data of size (batch_size, seq_len)
            target: expected next token of size (batch_size)
        """
        if len(self.available_indices) < batch_size:
            LOGGER.info(
                f"[DATA_LOADER] Only {len(self.available_indices)} tokens left. "
                f"Resetting available indices..."
            )
            self.available_indices = set(
                np.arange(self.token_num - self.seq_len).tolist()
            )

        # Sample batch_size indices from available_indices
        indices = np.random.choice(
            list(self.available_indices), size=batch_size, replace=False
        )
        self.available_indices -= set(indices)

        # Prepare data and target and move them to specified device
        data = np.stack([self.data[i : i + self.seq_len] for i in indices])
        data = torch.from_numpy(data).to(device).long()

        target = np.stack([self.data[i + 1 : i + self.seq_len + 1] for i in indices])
        target = torch.from_numpy(target).to(device).long()

        return data, target
