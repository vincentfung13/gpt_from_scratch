# Tokenization Module

This module implements a **Byte Pair Encoding (BPE)** tokenizer. The implementation emphasizes **efficient BPE training** through several optimizations over a naive brute-force algorithm, enabling training on large corpora (100M+ scale) in reasonable time.

---

## Brute-Force BPE vs. Optimized Implementation

A naive BPE training algorithm does the following at **each merge step**:

1. **Recount all pairs from scratch** — iterate over every pre-token and count all adjacent pairs.
2. **Scan all pre-tokens to apply the merge** — visit every unique pre-token to replace occurrences of the chosen pair, even when it does not contain that pair.
3. **Rebuild the pre-token dictionary** — construct a new `pre_token_counts` from the full corpus/state.

This yields per-step complexity on the order of **O(P × L)** for pair counting and **O(P × L)** for merging, where P = number of unique pre-tokens and L = average pre-token length. Over V merges (vocab growth), total cost is **O(V × P × L)**, which becomes prohibitive for large P (e.g., millions of unique pre-tokens).

The optimizations below reduce work per merge and keep total training cost much lower in practice.

---

## Optimizations for Efficient BPE Training

### 1. Pair-to-Pre-Token Reverse Index (Inverted Index)

**Brute force:** At each merge, scan all pre-tokens to find which ones contain the chosen pair. Cost: **O(P)** per step.

**Optimization:** Maintain a reverse index `pair_to_pre_tokens: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]]` that maps each pair to the set of pre-tokens that contain it.

- **Build once** during initialization when computing initial `pair_freq` and `pair_to_pre_tokens`.
- **At each merge:** Only process pre-tokens in `pair_to_pre_tokens[pair_to_merge]`.
- **After merging:** Update the index: remove the old pre-token from the sets of its pairs, and add the new merged pre-token to the sets of its new pairs.

**Effect:** Work per merge is proportional to the number of pre-tokens that **actually contain** the merged pair (often much smaller than P), instead of all P pre-tokens. This is the main lever for scaling to large P.

---

### 2. Incremental Pair Frequency Updates

**Brute force:** After each merge, recompute `pair_freq` by iterating over all pre-tokens and counting all adjacent pairs. Cost: **O(P × L)** per step.

**Optimization:** Update `pair_freq` incrementally while applying the merge in a single left-to-right pass over each affected pre-token:

- **Remove** the count of the merged pair `(a, b)` for each occurrence (batched per pre-token).
- For each merge of `(a, b)` into `merged_token`:
  - **Remove** `(prev_token, a)` and **add** `(prev_token, merged_token)` with the appropriate count.
  - **Remove** `(b, next_token)` and **add** `(merged_token, next_token)` with the appropriate count.

**Effect:** Pair counts stay correct without a full recount. Cost per merge is **O**(number of affected pre-tokens × length), not O(P × L).

---

### 3. In-Place Pre-Token Count Updates

**Brute force:** Each merge builds a brand-new `pre_token_counts` (or equivalent) by iterating over all pre-tokens and applying the merge, then replacing the old structure. This implies **O(P)** iterations and extra memory for the new structure.

**Optimization:**  
- Process only pre-tokens from `pair_to_pre_tokens[pair_to_merge]`.  
- For each such pre-token, remove it from `pre_token_counts` and add the merged (and possibly new) pre-tokens into `pre_token_counts` with the right counts.  
- No full pass over all P pre-tokens; only add/remove entries for pre-tokens that actually changed.

**Effect:** Fewer iterations and lower memory churn, which is important at 100M+ pre-token scales.

---

### 4. Bounded Scan for Finding the Best Pair

**Brute force:** To pick the pair with maximum frequency (and lexicographic tie-breaking), sort or scan the entire `pair_freq` structure. Cost: **O(N log N)** or **O(N)** per step, with N = number of distinct pairs.

**Optimization:** Use `pair_freq.most_common(min(1000, len(pair_freq)))` to obtain a limited number of top-frequency pairs, then among those with the **same** maximum frequency, choose the lexicographically largest in a single pass.  
This avoids a full sort of all pairs when the max is likely within a small top set.

**Effect:** Per-step selection cost is effectively **O(1)** when using a structure like `Counter` and only examining a bounded number of candidates, instead of O(N) or O(N log N) over the full pair set.

---

## Complexity Summary

| Operation                     | Brute Force        | Optimized                            |
|-----------------------------|--------------------|--------------------------------------|
| Find best pair              | O(N) or O(N log N) | O(1) with bounded `most_common`      |
| Apply merge (pre-tokens)    | O(P × L)           | O(A × L), A = pre-tokens with pair   |
| Update pair frequencies     | O(P × L)           | O(A × L), incremental                |
| Update pre-token structure  | O(P)               | O(A), in-place adds/removals         |

Here, **A** is the number of pre-tokens that contain the merged pair; typically A ≪ P for many merges, so the optimized implementation scales much better with P and V.

---

## Other Components

- **`BPETokenizer`** (`bpe.py`): Training, encoding, decoding, save/load.
- **`tokenization_utils.py`**: Pre-tokenization, chunking, and `encode_string` for applying merges.
- **`frequency_heap.py`**: Heap structure for max-frequency pair selection with lazy deletion (optional alternative to `Counter.most_common`).
- **`cfg.py`**: Pre-tokenization regex and related config.

For deeper bottleneck analysis and possible further improvements, see `BOTTLENECK_ANALYSIS.md`.

---

## Quick Reference

**Usage:**
```python
from gpt_from_scratch.tokenization.bpe import BPETokenizer

# Train
tokenizer = BPETokenizer()
tokenizer.train(input_path="data/train.txt", vocab_size=50000, save_dir=OUTPUT_DIR, special_tokens=["<|endoftext|>"])

# Load and use
tokenizer = BPETokenizer.from_file("OUTPUT_DIR/bpe_vocab.pkl", "OUTPUT_DIR/bpe_merges.pkl")
token_ids = tokenizer.encode("Hello, world!")
```

**Files:** `bpe_vocab.pkl`, `bpe_merges.pkl`, `special_tokens.txt` — see `BPETokenizer.save()` and `from_file()` in `bpe.py`.
