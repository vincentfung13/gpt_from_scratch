import regex as re
from typing import List, Dict, Tuple


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
    pre_tokenization_pattern: str,
):
    # Pre-tokenize, apply merges, then find indices
    tokens = []
    for match in re.finditer(pre_tokenization_pattern, input_string):
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


def pre_tokenize_text(
    text: str,
    pre_tokenization_pattern: str,
    special_tokens: List[str] = ["<|endoftext|>"],
) -> List[Tuple[bytes, ...]]:
    """
    Core pre-tokenization logic that works on a text string.

    Args:
        text: Input text string to pre-tokenize.
        pre_tokenization_pattern: regex pattern used to pre-tokenization the input text.
        special_tokens: List of special token strings to split on.

    Returns:
        List of pre tokens.
    """
    pre_tokens = []
    # Split by special tokens before pre-tokenizing to protect special token
    # Sort special tokens by length (longest first) to handle overlapping tokens correctly
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = r"|".join(re.escape(token) for token in sorted_special_tokens)
    for sub_chunk in re.split(pattern, text):
        # Run pre-tokenization and count each pre-token
        for match in re.finditer(pre_tokenization_pattern, sub_chunk):
            # Encode word to bytes, then split into individual bytes (byte-level BPE)
            word_bytes = match.group().encode("utf-8")
            pre_token_bytes_tuple = tuple(bytes([i]) for i in word_bytes)
            pre_tokens.append(pre_token_bytes_tuple)
    return pre_tokens
