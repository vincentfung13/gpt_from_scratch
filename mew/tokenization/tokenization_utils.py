import math
import regex as re
from typing import Dict, List, Tuple


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
    bpe_ranks: Dict[Tuple[bytes, bytes], int],
    pre_tokenization_pattern: str,
    cache: Dict[bytes, Tuple[bytes, ...]] | None = None,
):
    """Encode a string into token IDs using GPT-2-style byte-level BPE.

    This implements the standard BPE procedure:
    - pre-tokenize with the provided regex
    - for each pre-token (bytes), repeatedly merge the *lowest-rank* adjacent pair
      until no mergeable pairs remain
    """

    def _bpe(token_bytes: bytes) -> Tuple[bytes, ...]:
        if cache is not None:
            cached = cache.get(token_bytes)
            if cached is not None:
                return cached

        # Start from individual bytes.
        word: List[bytes] = [bytes([b]) for b in token_bytes]

        # Repeatedly merge the lowest-rank pair present.
        while len(word) >= 2:
            best_pair: Tuple[bytes, bytes] | None = None
            best_rank = math.inf
            for a, b in zip(word, word[1:]):
                rank = bpe_ranks.get((a, b))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            merged: List[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged

        out = tuple(word)
        if cache is not None:
            cache[token_bytes] = out
        return out

    tokens: List[int] = []
    for match in re.finditer(pre_tokenization_pattern, input_string):
        token_bytes = match.group().encode("utf-8")
        token_pieces = _bpe(token_bytes)
        tokens.extend(get_token_ind[p] for p in token_pieces)
    return tokens


def pre_tokenize_text(
    text: str,
    pre_tokenization_pattern: str,
    special_tokens: List[str] = ["<|endoftext|>"],
    preserve_special_tokens: bool = False,
) -> List[Tuple[bytes, ...]]:
    """
    Core pre-tokenization logic that works on a text string.

    Args:
        text: Input text string to pre-tokenize.
        pre_tokenization_pattern: regex pattern used to pre-tokenization the input text.
        special_tokens: List of special token strings to split on.
        preserve_special_tokens: Whether to keep special tokens as separate pre-tokens.

    Returns:
        List of pre tokens.
    """
    pre_tokens = []
    # Split by special tokens before pre-tokenizing to protect special token
    # Sort special tokens by length (longest first) to handle overlapping tokens correctly
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = r"|".join(re.escape(token) for token in sorted_special_tokens)
    if preserve_special_tokens and sorted_special_tokens:
        split_re = re.compile(f"({pattern})")
        for sub_chunk in split_re.split(text):
            if not sub_chunk:
                continue
            if sub_chunk in sorted_special_tokens:
                pre_tokens.append((sub_chunk.encode("utf-8"),))
                continue
            for match in re.finditer(pre_tokenization_pattern, sub_chunk):
                word_bytes = match.group().encode("utf-8")
                pre_token_bytes_tuple = tuple(bytes([i]) for i in word_bytes)
                pre_tokens.append(pre_token_bytes_tuple)
    else:
        for sub_chunk in re.split(pattern, text):
            for match in re.finditer(pre_tokenization_pattern, sub_chunk):
                word_bytes = match.group().encode("utf-8")
                pre_token_bytes_tuple = tuple(bytes([i]) for i in word_bytes)
                pre_tokens.append(pre_token_bytes_tuple)
    return pre_tokens
