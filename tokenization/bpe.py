from ast import Tuple
from typing import List, Dict

from tokenization.utils import find_chunk_boundaries


class BPETokenizer:
    def __init__(
        self, 
        vocab: Dict[int, bytes] = None, 
        merges: List[Tuple[bytes, bytes]] = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges

    def train(
        self, 
        input_path: str, 
        vocab_size: int, 
        special_tokens: List[str], 
        num_chunks: int = 8, 
        file_split_token: bytes = b"<|endoftext|>"
    ) -> None:
        # Init vocab
        # 1. Add special tokens
        # 2. Add 256 byte values
        vocab = {}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8") 
        for byte_ind in range(256):
            vocab[len(vocab)] = bytes([byte_ind])

        # Chunk up file
        with open(input_path, "rb") as fb:
            file_chunks = find_chunk_boundaries(fb, num_chunks, file_split_token) 

        # Run pre-tokenization and 

    def encode(self, text: str) -> List[int]:
        pass

    def decode(self, ids: List[int]) -> str:
        pass