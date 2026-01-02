from typing import List, Dict, Tuple
from collections import Counter

from gpt_from_scratch.tokenization.pre_tokenization import run_pre_tokenization


class BPETokenizer:
    def __init__(
        self, vocab: Dict[int, bytes] = None, merges: List[Tuple[bytes, bytes]] = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str] = ["<|endoftext|>"],
        num_chunks: int = 8,
        file_split_token: str = "<|endoftext|>",
    ) -> None:
        # Init vocab
        # 1. Add special tokens
        # 2. Add 256 byte values
        vocab = {}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")
        for byte_ind in range(256):
            vocab[len(vocab)] = bytes([byte_ind])

        # Run pre-tokenization
        pre_token_counts = run_pre_tokenization(
            input_file=input_path,
            num_chunks=num_chunks,
            chunk_split_special_token=file_split_token,
            special_tokens=special_tokens,
        )

        # Run training steps
        # Init merges (initially it's empty)
        merges = []
        for step in range(vocab_size - len(vocab)):
            # 1. Grab pair freq within each pre-tokens
            pair_freq = Counter()
            max_freq_pairs, max_freq = [], 0
            for pre_token, count in pre_token_counts.items():
                for pair in zip(pre_token, pre_token[1:]):
                    pair_freq[pair] += count
                    if pair_freq[pair] > max_freq:
                        max_freq = pair_freq[pair]
                        max_freq_pairs = [pair]
                    elif pair_freq[pair] == max_freq:
                        max_freq_pairs.append(pair)

            # 2. Find pair with max freq to merge
            # and update merges and vocab
            pair_to_merge = max(max_freq_pairs)
            new_token_id = len(vocab)
            merges.append(pair_to_merge)
            merged_token = pair_to_merge[0] + pair_to_merge[1]
            vocab[new_token_id] = merged_token

            # 3. Perform the merge on pre-tokens
            new_pre_token_counts = {}
            for pre_token, count in pre_token_counts.items():
                # Convert tuple to list for easier manipulation
                token_list = list(pre_token)

                # Merge all occurrences of the pair
                i = 0
                merged_list = []
                while i < len(token_list):
                    # Check if current and next byte form the merged pair
                    if (
                        i < len(token_list) - 1
                        and (token_list[i], token_list[i + 1]) == pair_to_merge
                    ):
                        # Merge: add the merged token bytes and skip the next byte
                        merged_list.append(merged_token)
                        i += 2
                    else:
                        # Keep the current byte
                        merged_list.append(token_list[i])
                        i += 1

                # Convert back to tuple and update counts
                merged_pre_token = tuple(merged_list)
                new_pre_token_counts[merged_pre_token] = (
                    new_pre_token_counts.get(merged_pre_token, 0) + count
                )
            pre_token_counts = new_pre_token_counts

        # Assign vocab and merges
        self.vocab = vocab
        self.merges = merges

    def encode(self, text: str) -> List[int]:
        pass

    def decode(self, ids: List[int]) -> str:
        pass


if __name__ == "__main__":
    bpe = BPETokenizer()
    bpe.train(
        "data/TineyStoriesV2-valid-samples.txt",
        500,
    )
