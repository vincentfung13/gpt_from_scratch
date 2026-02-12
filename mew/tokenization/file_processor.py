import os
import numpy as np
from collections import Counter
from typing import Iterator, Tuple, List, BinaryIO
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from tqdm import tqdm

from mew.tokenization.tokenization_utils import pre_tokenize_text
from mew import LOGGER

# Global tokenizer used by worker processes
_GLOBAL_TOKENIZER = None


def _init_tokenizer(tokenizer_path: str):
    from gpt_from_scratch.tokenization.bpe import BPETokenizer

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = BPETokenizer.from_file(
        vocab_filepath=os.path.join(tokenizer_path, "bpe_vocab.pkl"),
        merges_filepath=os.path.join(tokenizer_path, "bpe_merges.pkl"),
    )


def _tokenize_pre_token_worker(pre_token: Tuple[bytes, ...]):
    pre_token_str = b"".join(pre_token).decode("utf-8")
    return (pre_token, _GLOBAL_TOKENIZER.encode(pre_token_str))


class FileProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size_in_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.num_process_chunks = max(int(self.file_size_in_mb // 100), 8)
        self.pre_token_counts = None
        self.total_number_of_tokens = 0
        LOGGER.info(
            f"[FILE_PROCESSOR] File {self.file_path} has size {self.file_size_in_mb:.2f} MB, "
            f"using {self.num_process_chunks} chunks for pre-tokenization."
        )

    def get_pre_token_counts(
        self,
        chunk_split_special_token: str = "<|endoftext|>",
        special_tokens: List[str] = ["<|endoftext|>"],
    ) -> Counter[Tuple[bytes, ...], int]:
        if self.pre_token_counts is not None:
            return self.pre_token_counts

        self.pre_tokens_count = {}
        for pre_token in self.pre_tokenize(
            chunk_split_special_token=chunk_split_special_token,
            special_tokens=special_tokens,
        ):
            self.pre_tokens_count[pre_token] = (
                self.pre_tokens_count.get(pre_token, 0) + 1
            )
        return self.pre_tokens_count

    def pre_tokenize(
        self,
        num_workers=12,
        chunk_split_special_token: str = "<|endoftext|>",
        special_tokens: List[str] = ["<|endoftext|>"],
    ) -> Iterator[Tuple[bytes, ...]]:
        """
        Pre-tokenize the file.

        Args:
            raw_input: Either a file path (str) or a bytestream (bytes).
            chunk_split_special_token: Special token used to split the file into chunks safely.
            special_tokens: List of special token strings to split on.

        Returns:
            Iterator of pre-token byte tuples.
        """
        with open(self.file_path, "rb") as fb:
            boundaries = _find_chunk_boundaries(
                fb, self.num_process_chunks, chunk_split_special_token.encode("utf-8")
            )

        # Process chunks in parallel
        chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(
                total=len(chunk_pairs), desc="Pre-tokenizing chunks", unit="chunk"
            ) as pbar:
                window_size = min(num_workers, len(chunk_pairs))
                futures = {}
                for i in range(window_size):
                    start, end = chunk_pairs[i]
                    futures[i] = executor.submit(
                        _file_pre_tokenization_worker,
                        self.file_path,
                        start,
                        end,
                        special_tokens,
                    )
                next_to_submit = window_size
                for i in range(len(chunk_pairs)):
                    _pre_tokens = futures[i].result()
                    for pre_token in _pre_tokens:
                        yield pre_token
                    pbar.update(1)
                    if next_to_submit < len(chunk_pairs):
                        start, end = chunk_pairs[next_to_submit]
                        futures[next_to_submit] = executor.submit(
                            _file_pre_tokenization_worker,
                            self.file_path,
                            start,
                            end,
                            special_tokens,
                        )
                        next_to_submit += 1
                    del futures[i]

    def tokenize_file(
        self,
        tokenizer_path: str,
        output_path: str,
        num_workers: int = 32,
        chunk_split_special_token: str = "<|endoftext|>",
        special_tokens: List[str] = ["<|endoftext|>"],
    ) -> Iterator[Tuple[bytes, ...]]:
        # Dedup by pre-tokens
        pre_tokens_count = self.get_pre_token_counts(
            chunk_split_special_token=chunk_split_special_token,
            special_tokens=special_tokens,
        )
        pre_tokens = list(pre_tokens_count.keys())
        LOGGER.info(f"[FILE_PROCESSOR] Found {len(pre_tokens)} unique pre-tokens.")

        # Tokenize pre_tokens in parallel
        total_number_of_tokens = 0
        with Pool(
            processes=num_workers,
            initializer=_init_tokenizer,
            initargs=(tokenizer_path,),
        ) as pool:
            pre_token_to_tokens = {}
            results = list(
                tqdm(
                    pool.imap(_tokenize_pre_token_worker, pre_tokens),
                    total=len(pre_tokens),
                    desc="Running tokenization on deduped pre-tokens...",
                )
            )
            for pre_token, encoded_tokens in results:
                pre_token_to_tokens[pre_token] = encoded_tokens
                total_number_of_tokens += pre_tokens_count[pre_token] * len(
                    encoded_tokens
                )
        LOGGER.info(
            f"[FILE_PROCESSOR] Tokenization finished, found {total_number_of_tokens} tokens. "
            "Converting the file into tokens..."
        )

        arr = np.memmap(
            output_path, dtype=np.uint16, mode="w+", shape=(total_number_of_tokens,)
        )
        offset = 0
        processed_tokens = 0
        uint16_max = np.iinfo(np.uint16).max
        with tqdm(
            total=total_number_of_tokens, desc="Converting file to tokens", unit="token"
        ) as pbar:
            for ind, pre_token in enumerate(
                self.pre_tokenize(
                    chunk_split_special_token=chunk_split_special_token,
                    special_tokens=special_tokens,
                )
            ):
                if pre_token not in pre_token_to_tokens:
                    raise KeyError("Missing pre_token in mapping")

                tokens = pre_token_to_tokens[pre_token]
                if len(tokens) > 0:
                    # Checking for: 1. Offset exceeds allocated size; 2. Token id out of uint16 range
                    if offset + len(tokens) > total_number_of_tokens:
                        raise ValueError("Memmap slice exceeds allocated size")
                    if max(tokens) > uint16_max or min(tokens) < 0:
                        raise ValueError("Token id out of uint16 range")

                processed_tokens += len(tokens)
                arr[offset : offset + len(tokens)] = tokens
                offset += len(tokens)

                # Periodically flush to disk
                if ind % 100000 == 0:
                    arr.flush()

                pbar.update(len(tokens))

            # Final flush
            arr.flush()

        if (
            offset != total_number_of_tokens
            or processed_tokens != total_number_of_tokens
        ):
            raise ValueError("Memmap not fully filled or token count mismatch")

        LOGGER.info(
            f"[FILE_PROCESSOR] Saved {total_number_of_tokens} tokens to {output_path}."
        )


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # Refine chunk boundaries to ensure that each segment is not split across different chunks
    # by reading ahead by 4k bytes at a time and checking for the special token or EOF
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _file_pre_tokenization_worker(
    file_path: str,
    chunk_start_pos: int,
    chunk_end_pos: int,
    special_tokens: List[str] = ["<|endoftext|>"],
):
    # 1. Read file content
    with open(file_path, "rb") as file:
        file.seek(chunk_start_pos)
        chunk_content = file.read(chunk_end_pos - chunk_start_pos).decode(
            "utf-8", errors="ignore"
        )
    # 2. Use the core pre-tokenization logic
    return pre_tokenize_text(chunk_content, special_tokens)
