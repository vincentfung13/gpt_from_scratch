import os
import regex as re
from typing import BinaryIO, List, Dict
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


PRE_TOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def run_pre_tokenization(
    input_file: str,
    num_chunks: int = 12,
    chunk_split_special_token: bytes = b"<|endoftext|>",
    special_tokens: List[bytes] = ["<|endoftext|>"],
):
    # 1. Chunk up the input file
    # 2. Pre-tokenize per file
    # 3. Aggregate pre-token counts

    with open(input_file, "rb") as fb:
        boundaries = _find_chunk_boundaries(fb, num_chunks, chunk_split_special_token)

    # Process chunks in parallel
    pre_token_counter = Counter()
    chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

    with ProcessPoolExecutor() as executor:
        # Submit all chunk processing tasks
        future_to_chunk = {
            executor.submit(
                _pre_tokenization_worker, input_file, start, end, special_tokens
            ): (start, end)
            for start, end in chunk_pairs
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            _pre_token_counter = future.result()
            for pre_token, cnt in _pre_token_counter.items():
                pre_token_counter[pre_token] += cnt

    return pre_token_counter


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


def _pre_tokenization_worker(
    file_path: str,
    chunk_start_pos: int,
    chunk_end_pos: int,
    special_tokens: List[bytes] = ["<|endoftext|>"],
):
    # 1. Read file content
    with open(file_path, "rb") as file:
        file.seek(chunk_start_pos)
        chunk_content = file.read(chunk_end_pos - chunk_start_pos).decode(
            "utf-8", errors="ignore"
        )

    # 2. Split by special tokens before pre-tokenizing to protect special token
    pre_token_counter = Counter()
    for sub_chunk in re.split(r"|".join(special_tokens), chunk_content):
        # Run pre-tokenization and count each pre-token
        for match in re.finditer(PRE_TOKENIZATION_PATTERN, chunk_content):
            pre_token_bytes_tuple = tuple([ch.encode("utf-8") for ch in match.group()])
            pre_token_counter[pre_token_bytes_tuple] += 1

    return pre_token_counter


if __name__ == "__main__":
    print(_pre_tokenization_worker("data/TinyStoriesV2-GPT4-valid.txt", 0, 4096))
