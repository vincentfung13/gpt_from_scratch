import os
import numpy as np
from collections import Counter
from typing import Iterator, Tuple, List, BinaryIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from tqdm import tqdm

from gpt_from_scratch.tokenization.tokenization_utils import pre_tokenize_text
from gpt_from_scratch import LOGGER

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

            # Submit all chunk processing tasks
            future_to_chunk = [
                executor.submit(
                    _file_pre_tokenization_worker,
                    self.file_path,
                    start,
                    end,
                    special_tokens,
                )
                for start, end in chunk_pairs
            ]

            # Collect results as they complete
            with tqdm(
                total=len(future_to_chunk), desc="Pre-tokenizing chunks", unit="chunk"
            ) as pbar:
                for future in as_completed(future_to_chunk):
                    _pre_tokens = future.result()
                    for pre_token in _pre_tokens:
                        yield pre_token
                    pbar.update(1)

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

        LOGGER.info("[FILE_PROCESSOR] Converting file to tokens...")
        chunk_files = []
        buffer_tokens = []
        flush_every_tokens = 5000000
        part_idx = 0
        for pre_token in self.pre_tokenize(
            chunk_split_special_token=chunk_split_special_token,
            special_tokens=special_tokens,
        ):
            buffer_tokens.extend(pre_token_to_tokens[pre_token])
            if len(buffer_tokens) >= flush_every_tokens:
                part_path = f"{output_path}.part{part_idx}.bin"
                np.asarray(buffer_tokens, dtype=np.uint16).tofile(part_path)
                chunk_files.append(part_path)
                buffer_tokens.clear()
                part_idx += 1
                LOGGER.info(
                    f"[FILE_PROCESSOR] Saved {len(buffer_tokens)} tokens to {part_path}."
                )

        # Flush remaining tokens
        if buffer_tokens:
            part_path = f"{output_path}.part{part_idx}.bin"
            np.asarray(buffer_tokens, dtype=np.uint16).tofile(part_path)
            chunk_files.append(part_path)
            buffer_tokens.clear()
            LOGGER.info(
                f"[FILE_PROCESSOR] Saved {len(buffer_tokens)} tokens to {part_path}."
            )

        LOGGER.info(f"[FILE_PROCESSOR] Merging {len(chunk_files)} chunks...")
        total_tokens = 0
        for p in chunk_files:
            total_tokens += os.path.getsize(p) // np.dtype(np.uint16).itemsize
        arr = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        offset = 0
        for p in chunk_files:
            data = np.fromfile(p, dtype=np.uint16)
            arr[offset : offset + data.size] = data
            offset += data.size
        arr.flush()
        for p in chunk_files:
            try:
                os.remove(p)
            except OSError:
                pass
        LOGGER.info(f"[FILE_PROCESSOR] Saved {total_tokens} tokens to {output_path}.")


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


if __name__ == "__main__":
    tokenizer_path = "/mnt/bn/suhe-v6/zijian/cs336/owt_bpe_tokenizer"
    input_path = "/mnt/bn/suhe-v6/zijian/cs336/data/owt_train.txt"
    output_path = "/mnt/bn/suhe-v6/zijian/cs336/data/owt_train.tokens.uint16.npy"
    special_tokens = ["<|endoftext|>"]
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    fp = FileProcessor(file_path=input_path)
    fp.tokenize_file(
        tokenizer_path=tokenizer_path,
        special_tokens=special_tokens,
        output_path=output_path,
        num_workers=64
    )
