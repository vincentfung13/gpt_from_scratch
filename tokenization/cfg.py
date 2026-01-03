# Pre-tokenization pattern used for GPT-style BPE tokenizers.
# This should be a raw string (regex) and matches words, numbers, contractions, punctuation, and whitespaces.
PRE_TOKENIZATION_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
