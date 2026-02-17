from typing import Tuple, Union
import heapq


class FrequencyHeap:
    def __init__(self):
        self.heap = []
        self.items_to_remove = set()

    def head(self) -> Union[None, Tuple[Tuple[bytes, bytes]], int]:
        if len(self.heap) == 0:
            return None
        else:
            # Lazy pop
            while len(self.heap) > 0 and self.heap[0] in self.items_to_remove:
                self.items_to_remove.remove(self.heap[0])
                heapq.heappop(self.heap)
            if len(self.heap) == 0:
                return None
            return _convert_heap_item_to_pair_freq(self.heap[0])

    def pop(self) -> Union[None, Tuple[Tuple[bytes, bytes], int]]:
        if len(self.heap) == 0:
            raise ValueError("Popping from empty heap!")

        # Lazy pop
        while len(self.heap) > 0 and self.heap[0] in self.items_to_remove:
            self.items_to_remove.remove(self.heap[0])
            heapq.heappop(self.heap)

        # Remove current head
        if len(self.heap) == 0:
            return None
        else:
            return _convert_heap_item_to_pair_freq(heapq.heappop(self.heap))

    def remove(self, pair: Tuple[bytes, bytes], pair_freq: int) -> None:
        self.items_to_remove.add(_convert_pair_freq_to_heap_item(pair, pair_freq))

    def push(self, pair: Tuple[bytes, bytes], pair_freq: int) -> None:
        heap_item = _convert_pair_freq_to_heap_item(pair, pair_freq)
        heapq.heappush(self.heap, heap_item)


def _convert_heap_item_to_pair_freq(
    heap_item: Tuple[int, Tuple[int, ...], Tuple[int, ...]],
) -> Tuple[Tuple[bytes, bytes], int]:
    neg_freq, neg_bytes_0, neg_bytes_1 = heap_item
    # Convert tuple of integers back to bytes, negating to reverse the comparison
    # (we negated the bytes when storing to get lexicographically largest on ties)
    # bytes() constructor requires a list/iterable, not a generator directly
    return (
        (bytes([-b for b in neg_bytes_0]), bytes([-b for b in neg_bytes_1])),
        -neg_freq,
    )


def _convert_pair_freq_to_heap_item(
    pair: Tuple[bytes, bytes], pair_freq: int
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    # Use negative frequency for max-heap (heapq is min-heap)
    # Negate bytes to get lexicographically largest pair on ties:
    # When frequencies are equal, min-heap will compare negated byte tuples,
    # so smallest negated tuple = largest original pair
    return (-pair_freq, tuple(-b for b in pair[0]), tuple(-b for b in pair[1]))
