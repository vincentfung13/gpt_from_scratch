# Tokenization Module

This module implements a **Byte Pair Encoding (BPE)** tokenizer, which is a subword tokenization algorithm commonly used in modern language models like GPT-2, GPT-3, and GPT-4.

## Overview

The BPE tokenizer learns a vocabulary by iteratively merging the most frequent pairs of bytes/tokens in a training corpus. The resulting vocabulary consists of:
- **Special tokens** (e.g., `<|endoftext|>`) - assigned IDs starting from 256
- **256 base byte values** - the fundamental building blocks (IDs 0-255)
- **Learned merged tokens** - byte pairs that were merged during training (IDs 256+)

## Key Components

### `BPETokenizer` Class

The main tokenizer class that handles training, encoding, and decoding.

**Key Methods:**
- `train()`: Trains the tokenizer on a text corpus
- `encode()`: Converts text strings into token IDs
- `decode()`: Converts token IDs back into text strings
- `save()`: Saves the trained tokenizer to disk
- `from_file()`: Loads a trained tokenizer from disk

### `tokenization_utils.py`

Utility functions that support the tokenization process:

- `run_pre_tokenization()`: Pre-tokenizes input text into chunks for parallel processing
- `encode_string()`: Applies BPE merges to encode a string into token IDs
- `_pre_tokenize_text()`: Core pre-tokenization logic using regex patterns
- `_find_chunk_boundaries()`: Finds safe boundaries for parallel file processing

### `cfg.py`

Configuration file containing the pre-tokenization regex pattern used to split text into words, numbers, contractions, punctuation, and whitespace.

## Key Implementation Details

### 1. Byte-Level BPE

This implementation uses **byte-level BPE**, meaning:
- All text is first encoded to UTF-8 bytes
- The base vocabulary consists of all 256 possible byte values
- Merges operate on byte pairs, ensuring the tokenizer can handle any Unicode text without an out-of-vocabulary (OOV) problem

### 2. Pre-Tokenization

Before applying BPE merges, text is pre-tokenized using a regex pattern that matches:
- Words (including contractions like `'ll`, `'ve`, `'re`)
- Numbers
- Punctuation
- Whitespace

This pre-tokenization step ensures that merges only occur within words, not across word boundaries, which helps preserve linguistic structure.

### 3. Training Process

The training algorithm follows these steps:

1. **Initialize vocabulary**: Start with 256 base bytes + special tokens
2. **Pre-tokenize corpus**: Split text into chunks and count occurrences of each pre-token
3. **Iterative merging**:
   - Count frequency of all adjacent byte pairs in pre-tokens
   - Find the most frequent pair (with lexicographic tie-breaking)
   - Merge the pair into a new token
   - Update all pre-tokens by replacing occurrences of the pair with the merged token
   - Repeat until target vocabulary size is reached

4. **Save results**: Store vocabulary, merges, and special tokens to disk

### 4. Encoding Process

When encoding text:

1. **Split by special tokens**: Text is first split by special tokens (sorted by length, longest first) to handle overlapping tokens correctly
2. **Pre-tokenize**: Each segment is pre-tokenized using the regex pattern
3. **Apply merges**: For each pre-token:
   - Start with individual bytes
   - Apply all learned merges in order (greedy left-to-right)
   - Convert merged tokens to token IDs using the vocabulary

### 5. Decoding Process

Decoding is straightforward:
- Look up each token ID in the vocabulary to get its byte representation
- Concatenate all bytes
- Decode as UTF-8 (with error handling for invalid sequences)

### 6. Special Token Handling

Special tokens are handled carefully:
- They are added to the vocabulary during initialization
- During encoding, text is split by special tokens first to preserve them
- Special tokens are matched longest-first to handle overlapping cases (e.g., `"<|endoftext|><|endoftext|>"`)

### 7. Parallel Processing

For efficient training on large corpora:
- The input file is split into chunks at safe boundaries (using special tokens)
- Each chunk is processed in parallel using `ProcessPoolExecutor`
- Pre-token counts are aggregated across all chunks
- This allows training on large files without loading everything into memory

### 8. File Chunking Strategy

The chunking algorithm:
- Divides the file into roughly equal-sized chunks
- Refines boundaries by searching for special tokens in 4KB windows
- Ensures chunks don't split text in the middle of a segment
- May produce fewer chunks than requested if boundaries overlap

## Usage Example

```python
from gpt_from_scratch.tokenization.bpe import BPETokenizer

# Train a new tokenizer
tokenizer = BPETokenizer()
tokenizer.train(
    input_path="data/train.txt",
    vocab_size=50000,
    special_tokens=["<|endoftext|>"]
)

# Load a trained tokenizer
tokenizer = BPETokenizer.from_file(
    vocab_filepath="tokenizer_checkpoints/bpe_vocab.pkl",
    merges_filepath="tokenizer_checkpoints/bpe_merges.pkl",
    special_tokens=["<|endoftext|>"]
)

# Encode text
token_ids = tokenizer.encode("Hello, world! <|endoftext|>")

# Decode tokens
text = tokenizer.decode(token_ids)
```

## File Format

The tokenizer saves three files:
- `bpe_vocab.pkl`: Pickle file containing a dictionary mapping token IDs to byte sequences
- `bpe_merges.pkl`: Pickle file containing a list of byte pairs that were merged (in order)
- `special_tokens.txt`: Plain text file with one special token per line

## Design Decisions

1. **Byte-level encoding**: Ensures no OOV tokens, but may produce longer sequences than character-level or word-level tokenization
2. **Greedy left-to-right merging**: During encoding, merges are applied greedily from left to right, matching the training process
3. **Pre-tokenization**: Using regex patterns ensures merges respect word boundaries, improving token quality
4. **Parallel processing**: Chunk-based parallelization allows training on large files efficiently
5. **Pickle format**: Binary format for efficient storage and loading of byte sequences

