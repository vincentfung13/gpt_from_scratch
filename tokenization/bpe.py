import pickle
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter

from gpt_from_scratch.tokenization.pre_tokenization import run_pre_tokenization


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation.

    This tokenizer learns a vocabulary by iteratively merging the most frequent
    pairs of bytes/tokens in the training corpus. The vocabulary consists of:
    - Special tokens (e.g., <|endoftext|>)
    - 256 base byte values
    - Learned merged tokens from the training process
    """

    def __init__(
        self,
        vocab: Optional[Dict[int, bytes]] = None,
        merges: Optional[List[Tuple[bytes, bytes]]] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the BPE tokenizer.

        Args:
            vocab: Dictionary mapping token IDs to byte sequences. If None,
                   tokenizer must be trained before use.
            merges: List of byte pairs that were merged during training, in order
                    of merging. If None, tokenizer must be trained before use.
            special_tokens: List of special token strings (e.g., ["<|endoftext|>"]).
                           If None, defaults to empty list.
        """
        self.vocab: Dict[int, bytes] = vocab if vocab is not None else {}
        self.merges: List[Tuple[bytes, bytes]] = merges if merges is not None else []
        self.special_tokens: List[str] = (
            special_tokens if special_tokens is not None else []
        )

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """
        Load a trained tokenizer from pickle files.

        Args:
            vocab_filepath: Path to the vocabulary pickle file.
            merges_filepath: Path to the merges pickle file.
            special_tokens: Optional list of special tokens. If None, will attempt
                           to load from special_tokens.txt in the same directory.

        Returns:
            BPETokenizer instance loaded from files.
        """
        # Load vocab from pickle file (binary mode required for pickle)
        with open(vocab_filepath, "rb") as f:
            vocab: Dict[int, bytes] = pickle.load(f)

        # Load merges from pickle file (binary mode required for pickle)
        with open(merges_filepath, "rb") as f:
            merges: List[Tuple[bytes, bytes]] = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: List[str] = ["<|endoftext|>"],
        num_chunks: int = 8,
        file_split_token: str = "<|endoftext|>",
    ) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        The training process:
        1. Initializes vocabulary with special tokens and 256 base byte values
        2. Pre-tokenizes the input corpus into chunks
        3. Iteratively merges the most frequent byte pairs until vocab_size is reached
        4. Saves the trained tokenizer to disk

        Args:
            input_path: Path to the training text file.
            vocab_size: Target vocabulary size (includes special tokens and base bytes).
            special_tokens: List of special token strings to include in vocabulary.
            num_chunks: Number of chunks to split the input file for parallel processing.
            file_split_token: Special token used to split the file into chunks safely.
        """
        # Initialize vocabulary
        # Step 1: Add all 256 possible byte values as base tokens (IDs 0-255)
        vocab: Dict[int, bytes] = {}
        for byte_ind in range(256):
            vocab[byte_ind] = bytes([byte_ind])

        # Step 2: Add special tokens (they get IDs 256+)
        for i, special_token in enumerate(special_tokens):
            token_id = 256 + i
            vocab[token_id] = special_token.encode("utf-8")

        # Run pre-tokenization to split text into chunks and count occurrences
        # Returns a Counter mapping pre-token tuples (of bytes) to their counts
        pre_token_counts: Counter[Tuple[bytes, ...]] = run_pre_tokenization(
            input_file=input_path,
            num_chunks=num_chunks,
            chunk_split_special_token=file_split_token,
            special_tokens=special_tokens,
        )

        # Run BPE training: iteratively merge most frequent pairs
        # Initialize merges list (initially empty, will be populated during training)
        merges: List[Tuple[bytes, bytes]] = []

        # Continue merging until we reach the target vocabulary size
        for step in range(vocab_size - len(vocab)):
            # Step 1: Count frequency of all adjacent byte pairs in pre-tokens
            pair_freq: Counter[Tuple[bytes, bytes]] = Counter()
            for pre_token, count in pre_token_counts.items():
                # Skip pre-tokens with less than 2 elements (can't form pairs)
                if len(pre_token) < 2:
                    continue
                # Count all adjacent pairs in this pre-token, weighted by count
                for pair in zip(pre_token, pre_token[1:]):
                    pair_freq[pair] += count

            # Step 2: Find the pair with maximum frequency to merge
            # If multiple pairs have the same max frequency, choose lexicographically largest
            if not pair_freq:
                # No more pairs to merge, stop early
                break
            max_freq: int = max(pair_freq.values())
            max_freq_pairs: List[Tuple[bytes, bytes]] = [
                pair for pair, freq in pair_freq.items() if freq == max_freq
            ]
            pair_to_merge: Tuple[bytes, bytes] = max(max_freq_pairs)

            # Create new token by concatenating the merged pair
            new_token_id: int = len(vocab)
            merges.append(pair_to_merge)
            merged_token: bytes = pair_to_merge[0] + pair_to_merge[1]
            vocab[new_token_id] = merged_token

            # Step 3: Perform the merge on all pre-tokens
            # Update pre_token_counts by replacing all occurrences of the pair with the merged token
            new_pre_token_counts: Dict[Tuple[bytes, ...], int] = {}
            for pre_token, count in pre_token_counts.items():
                # Convert tuple to list for easier manipulation
                token_list: List[bytes] = list(pre_token)

                # Merge all occurrences of the pair (greedy left-to-right)
                i: int = 0
                merged_list: List[bytes] = []
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
                        # Keep the current byte/token as-is
                        merged_list.append(token_list[i])
                        i += 1

                # Convert back to tuple (for use as dictionary key) and update counts
                merged_pre_token: Tuple[bytes, ...] = tuple(merged_list)
                new_pre_token_counts[merged_pre_token] = (
                    new_pre_token_counts.get(merged_pre_token, 0) + count
                )
            pre_token_counts = new_pre_token_counts

        # Assign vocab and merges to instance variables
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # Save the trained tokenizer to disk
        self.save()

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of token IDs.

        This method should apply the BPE merges learned during training to convert
        text into token IDs. Currently not implemented.

        Args:
            text: Input text string to encode.

        Returns:
            List of token IDs corresponding to the encoded text.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        pass

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        This method should convert token IDs back to their byte representations
        and decode them as UTF-8 text. Currently not implemented.

        Args:
            ids: List of token IDs to decode.

        Returns:
            Decoded text string.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        pass

    def save(self, root_dir: str = "tokenizer_checkpoints") -> None:
        """
        Save the trained tokenizer to disk.

        Saves three files:
        - bpe_vocab.pkl: Vocabulary dictionary (token ID -> bytes)
        - bpe_merges.pkl: List of merge operations (list of byte pairs)
        - special_tokens.txt: List of special tokens (one per line)

        Args:
            root_dir: Directory where tokenizer files will be saved.
                     Will be created if it doesn't exist.
        """
        # Create output directory if it doesn't exist
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # Define output file paths
        vocab_out_path: str = os.path.join(root_dir, "bpe_vocab.pkl")
        merges_out_path: str = os.path.join(root_dir, "bpe_merges.pkl")
        special_tokens_path: str = os.path.join(root_dir, "special_tokens.txt")

        # Save vocabulary dictionary as pickle file
        with open(vocab_out_path, "wb") as vocab_f:
            pickle.dump(self.vocab, vocab_f)

        # Save merges list as pickle file
        with open(merges_out_path, "wb") as merges_f:
            pickle.dump(self.merges, merges_f)

        # Save special tokens as plain text (one per line)
        with open(special_tokens_path, "w") as st_f:
            for special_token in self.special_tokens:
                st_f.write(special_token + "\n")


if __name__ == "__main__":
    # Example usage: train a BPE tokenizer on a text corpus
    bpe = BPETokenizer()
    bpe.train(
        "data/TineyStoriesV2-valid-samples.txt",
        1000,
    )
