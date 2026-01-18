import os
import regex as re
from typing import BinaryIO, List, Union, Dict, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from gpt_from_scratch.tokenization import cfg


def find_most_freq_pair_to_merge(
    pair_freq: Dict[Tuple[bytes, bytes], int],
) -> Tuple[Tuple[bytes, bytes], int]:
    # If multiple pairs have the same max frequency, choose lexicographically largest
    # Optimization: Use most_common() with a reasonable limit, then find lexicographically
    # largest among pairs with max frequency in a single pass

    most_common_iter = pair_freq.most_common(min(1000, len(pair_freq)))
    if not most_common_iter:
        return

    # Get first pair to establish max_freq
    first_pair, max_freq = most_common_iter[0]
    pair_to_merge: Tuple[bytes, bytes] = first_pair

    # Iterate through remaining pairs with same frequency, tracking lexicographically largest
    for pair, freq in most_common_iter[1:]:
        if freq < max_freq:
            # Since most_common returns in descending order, we can stop here
            break
        # Update if this pair is lexicographically larger
        if pair > pair_to_merge:
            pair_to_merge = pair

    return pair_to_merge, max_freq


def encode_string(
    input_string: str,
    get_token_ind: Dict[bytes, int],
    merges: List[tuple[bytes, bytes]],
):
    # Pre-tokenize, apply merges, then find indices
    tokens = []
    for match in re.finditer(cfg.PRE_TOKENIZATION_PATTERN, input_string):
        # Encode word to bytes, then split into individual bytes (byte-level BPE)
        word_bytes = match.group().encode("utf-8")
        token_list = [bytes([i]) for i in word_bytes]

        # Apply merges in-order
        for pair_to_merge in merges:
            ind = 0
            new_token_list = []
            while ind < len(token_list):
                if (
                    ind < len(token_list) - 1
                    and (token_list[ind], token_list[ind + 1]) == pair_to_merge
                ):
                    new_token_list.append(pair_to_merge[0] + pair_to_merge[1])
                    ind += 2
                else:
                    new_token_list.append(token_list[ind])
                    ind += 1
            token_list = new_token_list

        # Add to token list
        tokens += [get_token_ind[token] for token in token_list]
    return tokens


def run_pre_tokenization(
    raw_input: Union[str, bytes],
    num_chunks: int = 12,
    chunk_split_special_token: str = "<|endoftext|>",
    special_tokens: List[str] = ["<|endoftext|>"],
):
    """
    Pre-tokenize input from either a file path or a bytestream.

    Args:
        raw_input: Either a file path (str) or a bytestream (bytes).
        num_chunks: Number of chunks to split the raw_input for parallel processing.
                    Only used when raw_input is a file path.
        chunk_split_special_token: Special token used to split the file into chunks safely.
                                  Only used when raw_input is a file path.
        special_tokens: List of special token strings to split on.

    Returns:
        Counter mapping pre-token byte tuples to their counts.
    """
    # Handle bytestream input
    if isinstance(raw_input, bytes):
        # Decode bytes to string and pre-tokenize directly
        text = raw_input.decode("utf-8", errors="ignore")
        return _pre_tokenize_text(text, special_tokens)

    # Handle file path input (original behavior)
    # 1. Chunk up the input file
    # 2. Pre-tokenize per file
    # 3. Aggregate pre-token counts
    with open(raw_input, "rb") as fb:
        boundaries = _find_chunk_boundaries(
            fb, num_chunks, chunk_split_special_token.encode("utf-8")
        )

    # Process chunks in parallel
    pre_token_counter = Counter()
    chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

    with ProcessPoolExecutor() as executor:
        # Submit all chunk processing tasks
        future_to_chunk = {
            executor.submit(
                _file_pre_tokenization_worker, raw_input, start, end, special_tokens
            ): (start, end)
            for start, end in chunk_pairs
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            _pre_token_counter = future.result()
            for pre_token, cnt in _pre_token_counter.items():
                pre_token_counter[pre_token] += cnt

    return pre_token_counter


def _pre_tokenize_text(
    text: str,
    special_tokens: List[str] = ["<|endoftext|>"],
) -> Counter:
    """
    Core pre-tokenization logic that works on a text string.

    Args:
        text: Input text string to pre-tokenize.
        special_tokens: List of special token strings to split on.

    Returns:
        Counter mapping pre-token byte tuples to their counts.
    """
    pre_token_counter = Counter()
    # Split by special tokens before pre-tokenizing to protect special token
    # Sort special tokens by length (longest first) to handle overlapping tokens correctly
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = r"|".join(re.escape(token) for token in sorted_special_tokens)
    for sub_chunk in re.split(pattern, text):
        # Run pre-tokenization and count each pre-token
        for match in re.finditer(cfg.PRE_TOKENIZATION_PATTERN, sub_chunk):
            # Encode word to bytes, then split into individual bytes (byte-level BPE)
            word_bytes = match.group().encode("utf-8")
            pre_token_bytes_tuple = tuple(bytes([i]) for i in word_bytes)
            pre_token_counter[pre_token_bytes_tuple] += 1

    return pre_token_counter


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
    return _pre_tokenize_text(chunk_content, special_tokens)


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
