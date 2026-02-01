from collections.abc import Iterator
import time
import pickle
import os
import regex as re
from tqdm import tqdm
from typing import Iterable, List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict

from gpt_from_scratch.tokenization.tokenization_utils import (
    find_most_freq_pair_to_merge,
    encode_string,
)
from gpt_from_scratch import LOGGER


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
        self.get_token_ind: Dict[bytes, int] = {
            token: token_ind for token_ind, token in self.vocab.items()
        }
        self.merges: List[Tuple[bytes, bytes]] = merges if merges is not None else []
        self.special_tokens: List[str] = (
            special_tokens if special_tokens is not None else []
        )

        # Add special token into vocab
        for special_token in self.special_tokens:
            special_token = special_token.encode("utf-8")
            if special_token not in self.get_token_ind:
                new_token_ind = len(self.vocab)
                self.vocab[new_token_ind] = special_token
                self.get_token_ind[special_token] = new_token_ind

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = ["<|endoftext|>"],
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
        save_dir: Union[None, str],
        special_tokens: List[str] = ["<|endoftext|>"],
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
            save_dir: Directory to save the pre-trained tokenizer, set to None to disable saving (e.g for testing).
            special_tokens: List of special token strings to include in vocabulary.
            file_split_token: Special token used to split the file into chunks safely.
        """
        LOGGER.info(f"Starting BPE training on {input_path}")
        LOGGER.info(
            f"Target vocabulary size: {vocab_size}, Special tokens: {special_tokens}"
        )

        # Initialize vocabulary
        # Step 1: Add all 256 possible byte values as base tokens (IDs 0-255)
        vocab: Dict[int, bytes] = {}
        for byte_ind in range(256):
            vocab[byte_ind] = bytes([byte_ind])

        # Step 2: Add special tokens (they get IDs 256+)
        for i, special_token in enumerate(special_tokens):
            token_id = 256 + i
            vocab[token_id] = special_token.encode("utf-8")

        initial_vocab_size = len(vocab)
        LOGGER.info(
            f"Initialized vocabulary with {initial_vocab_size} tokens (256 base bytes + {len(special_tokens)} special tokens)"
        )

        # Run pre-tokenization to split text into chunks and count occurrences
        # Returns a Counter mapping pre-token tuples (of bytes) to their counts
        from gpt_from_scratch.tokenization.file_processor import FileProcessor

        LOGGER.info("Running pre-tokenization...")
        fp = FileProcessor(file_path=input_path)
        pre_token_counts: Counter[Tuple[bytes, ...]] = fp.get_pre_token_counts(
            chunk_split_special_token=file_split_token,
            special_tokens=special_tokens,
        )
        LOGGER.info(
            f"Pre-tokenization complete. Found {len(pre_token_counts)} unique pre-tokens"
        )

        # Initialize pre-token counts
        pair_freq: Counter[Tuple[bytes, bytes]] = Counter()
        # Build reverse index: map each pair to set of pre-tokens containing it
        # This allows us to only process relevant pre-tokens during merging
        pair_to_pre_tokens: Dict[Tuple[bytes, bytes], set] = defaultdict(set)
        for pre_token, count in pre_token_counts.items():
            # Skip pre-tokens with less than 2 elements (can't form pairs)
            if len(pre_token) >= 2:
                # Count all adjacent pairs in this pre-token, weighted by count
                for pair in zip(pre_token, pre_token[1:]):
                    pair_freq[pair] += count
                    # Build reverse index
                    pair_to_pre_tokens[pair].add(pre_token)

        # Run BPE training: iteratively merge most frequent pairs
        # Initialize merges list (initially empty, will be populated during training)
        merges: List[Tuple[bytes, bytes]] = []

        total_steps = vocab_size - len(vocab)
        LOGGER.info(
            f"Starting BPE merge process. Will perform {total_steps} merge steps to reach vocab size {vocab_size}"
        )

        # Continue merging until we reach the target vocabulary size
        for step in tqdm(list(range(total_steps)), desc="Running merges..."):
            start_ts = time.time()

            # Step 1: Find the pair with maximum frequency to merge
            pair_to_merge, max_freq = find_most_freq_pair_to_merge(pair_freq=pair_freq)
            merged_token: bytes = pair_to_merge[0] + pair_to_merge[1]

            # Step 2: Create new token in the merges and vocab
            new_token_id: int = len(vocab)
            merges.append(pair_to_merge)
            vocab[new_token_id] = merged_token
            found_pair_to_merge_ts = time.time()

            # Step 3: Perform the merge on all pre-tokens and update pair_freq incrementally

            # Optimization: Only process pre-tokens that contain the pair being merged.
            # Note: making a copy here to avoid mutation during iteration
            pre_tokens_to_process = list(pair_to_pre_tokens.get(pair_to_merge, set()))
            processed_pre_tokens = set()

            # Process pre-tokens that contain the pair
            new_pre_token_counts: Dict[Tuple[bytes, ...], int] = {}
            for pre_token in pre_tokens_to_process:
                count = pre_token_counts.get(pre_token, 0)
                processed_pre_tokens.add(pre_token)

                # Skip pre-tokens that are too short to contain a pair
                if len(pre_token) < 2:
                    new_pre_token_counts[pre_token] = (
                        new_pre_token_counts.get(pre_token, 0) + count
                    )
                    continue

                # Perform merge and update pair_freq in a single pass (greedy left-to-right)
                token_list: List[bytes] = list(pre_token)
                merged_list: List[bytes] = []
                pre_token_ind: int = 0
                changed = False  # Track if we made any merges

                while pre_token_ind < len(token_list):
                    # Check if current and next byte form the merged pair
                    if (
                        pre_token_ind < len(token_list) - 1
                        and token_list[pre_token_ind] == pair_to_merge[0]
                        and token_list[pre_token_ind + 1] == pair_to_merge[1]
                    ):
                        changed = True

                        # Update pair_freq before merging:
                        pair_freq[pair_to_merge] -= count
                        if pair_freq[pair_to_merge] <= 0:
                            del pair_freq[pair_to_merge]

                        # 2. If there's a token before pair_to_merge[0], update pairs involving it
                        if len(merged_list):
                            prev_token = merged_list[-1]
                            # Remove (prev_token, pair_to_merge[0]) pair (if it exists in pair_freq)
                            old_pair_before = (prev_token, pair_to_merge[0])
                            old_freq_before = pair_freq.get(old_pair_before, 0)
                            if old_freq_before > 0:
                                new_freq = old_freq_before - count
                                if new_freq > 0:
                                    pair_freq[old_pair_before] = new_freq
                                else:
                                    del pair_freq[old_pair_before]
                            # Add (prev_token, merged_token) pair
                            new_pair_before = (prev_token, merged_token)
                            pair_freq[new_pair_before] = (
                                pair_freq.get(new_pair_before, 0) + count
                            )

                        # 3. If there's a token after pair_to_merge[1], update pairs involving it
                        if pre_token_ind + 2 < len(token_list):
                            next_token = token_list[pre_token_ind + 2]
                            # Remove (pair_to_merge[1], next_token) pair
                            old_pair_after = (pair_to_merge[1], next_token)
                            old_freq_after = pair_freq.get(old_pair_after, 0)
                            if old_freq_after > 0:
                                new_freq = old_freq_after - count
                                if new_freq > 0:
                                    pair_freq[old_pair_after] = new_freq
                                else:
                                    del pair_freq[old_pair_after]
                            # Add (merged_token, next_token) pair
                            new_pair_after = (merged_token, next_token)
                            pair_freq[new_pair_after] = (
                                pair_freq.get(new_pair_after, 0) + count
                            )

                        # Perform the merge: add the merged token and skip the next byte
                        merged_list.append(merged_token)
                        pre_token_ind += 2
                    else:
                        # Keep the current byte/token as-is
                        merged_list.append(token_list[pre_token_ind])
                        pre_token_ind += 1

                # Update reverse index: remove old pre-token from all its pairs
                # and add merged pre-token to all its pairs
                if changed:
                    merged_pre_token: Tuple[bytes, ...] = tuple(merged_list)
                    if len(pre_token) >= 2:
                        for old_pair in zip(pre_token, pre_token[1:]):
                            if old_pair in pair_to_pre_tokens:
                                pair_to_pre_tokens[old_pair].discard(pre_token)
                                if not pair_to_pre_tokens[old_pair]:
                                    del pair_to_pre_tokens[old_pair]
                    # Add merged pre-token to reverse index for all its pairs
                    if len(merged_pre_token) >= 2:
                        for new_pair in zip(merged_pre_token, merged_pre_token[1:]):
                            if new_pair not in pair_to_pre_tokens:
                                pair_to_pre_tokens[new_pair] = set()
                            pair_to_pre_tokens[new_pair].add(merged_pre_token)
                else:
                    merged_pre_token = pre_token  # Reuse original tuple
                new_pre_token_counts[merged_pre_token] = (
                    new_pre_token_counts.get(merged_pre_token, 0) + count
                )
            pre_token_merge_ts = time.time()

            # In-place update: only remove processed and add merged. Avoids iterating over
            # all unique pre-tokens (O(|pre_token_counts|)) each merge—critical for 100M+ scale.
            for processed_pre_token in processed_pre_tokens:
                pre_token_counts.pop(processed_pre_token)
            for new_pre_token, count in new_pre_token_counts.items():
                pre_token_counts[new_pre_token] = (
                    pre_token_counts.get(new_pre_token, 0) + count
                )

            # Clean up reverse index: remove the merged pair if it has no more pre-tokens
            if pair_to_merge in pair_to_pre_tokens:
                del pair_to_pre_tokens[pair_to_merge]
            clean_up_ts = time.time()

            if step > 0 and step % 10 == 0:
                LOGGER.info(
                    "Merge step %d: merged pair %s + %s (freq=%d). Profile: find_pair=%.4fs merge_pretokens=%.4fs cleanup=%.4fs",
                    step + 1,
                    repr(pair_to_merge[0]),
                    repr(pair_to_merge[1]),
                    max_freq,
                    found_pair_to_merge_ts - start_ts,
                    pre_token_merge_ts - found_pair_to_merge_ts,
                    clean_up_ts - pre_token_merge_ts,
                )

        # Assign vocab and merges to instance variables
        self.vocab = vocab
        self.merges = merges
        self.get_token_ind: Dict[bytes, int] = {
            token: token_ind for token_ind, token in self.vocab.items()
        }
        self.special_tokens = special_tokens

        # Save the trained tokenizer to disk
        if save_dir is not None:
            self.save(save_dir=save_dir)

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of token IDs.

        This method should apply the BPE merges learned during training to convert
        text into token IDs.

        Args:
            text: Input text string to encode.

        Returns:
                List of token IDs corresponding to the encoded text.
        """
        # Run pre-tokenization
        # 1. Split by special tokens and also preserve them
        # Sort special tokens by length (longest first) to handle overlapping tokens correctly
        # This ensures that longer special tokens (e.g., "<|endoftext|><|endoftext|>") are
        # matched before shorter ones (e.g., "<|endoftext|>") when they overlap
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        tokens = []

        # If there are no special tokens, process the whole text directly
        if not sorted_special_tokens:
            return encode_string(
                input_string=text, get_token_ind=self.get_token_ind, merges=self.merges
            )

        # If there are special tokens, split by them first
        pattern = r"|".join(re.escape(token) for token in sorted_special_tokens)
        split_re = re.compile(f"({pattern})")
        tokens = []
        for sub_chunk in split_re.split(text):
            # Skip empty strings from split
            if not sub_chunk:
                continue

            # Check if this chunk is a special token (directly adding token id)
            if sub_chunk in self.special_tokens:
                tokens.append(self.get_token_ind[sub_chunk.encode("utf-8")])
                continue

            # Pre-tokenize and encode the string
            tokens += encode_string(
                input_string=sub_chunk,
                get_token_ind=self.get_token_ind,
                merges=self.merges,
            )

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for input_texts in iterable:
            for token_ind in self.encode(input_texts):
                yield token_ind

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        This method should convert token IDs back to their byte representations
        and decode them as UTF-8 text. Currently not implemented.

        Args:
            ids: List of token IDs to decode.

        Returns:
            Decoded text string.
        """
        code_points = []
        for _id in ids:
            code_points += list(self.vocab[_id])
        return bytes(code_points).decode("utf-8", errors="replace")

    def save(self, save_dir: str = "tokenizer_checkpoints") -> None:
        """
        Save the trained tokenizer to disk.

        Saves three files:
        - bpe_vocab.pkl: Vocabulary dictionary (token ID -> bytes)
        - bpe_merges.pkl: List of merge operations (list of byte pairs)
        - special_tokens.txt: List of special tokens (one per line)

        Args:
            save_dir: Directory where tokenizer files will be saved.
                     Will be created if it doesn't exist.
        """
        # Create output directory if it doesn't exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            LOGGER.debug(f"Created directory: {save_dir}")

        # Define output file paths
        vocab_out_path: str = os.path.join(save_dir, "bpe_vocab.pkl")
        merges_out_path: str = os.path.join(save_dir, "bpe_merges.pkl")
        special_tokens_path: str = os.path.join(save_dir, "special_tokens.txt")

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

        LOGGER.debug(
            f"Saved tokenizer files to {save_dir}: vocab ({len(self.vocab)} tokens), merges ({len(self.merges)} merges), special_tokens ({len(self.special_tokens)} tokens)"
        )


if __name__ == "__main__":
    # Example usage: train a BPE tokenizer on a text corpus
    bpe = BPETokenizer()
    bpe.train(
        input_path="/mnt/bn/suhe-v6/zijian/cs336/data/owt_train.txt",
        save_dir="/mnt/bn/suhe-v6/zijian/cs336/owt_bpe",
        vocab_size=32000,
    )
    print(
        bpe.decode(
            bpe.encode(
                "Hello <|endoftext|><|endoftext|> world my name is deep shit <|endoftext|>"
            )
        )
    )
