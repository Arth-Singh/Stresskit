# Lens baseline comparison — Qwen/Qwen3.5-4B

Hit criterion: every intermediate of an item at rank <= 5 (word-like mask, all fitted layers, position -1). `found@100`: intermediate appears anywhere in the top-100.

## lens-eval-association

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.0392 | 0.2157 | 15 |
| logit | 0.0 | 0.1471 | 25 |
| tuned | 0.0 | 0.1176 | 41 |

## lens-eval-multihop

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.3118 | 0.5806 | 4 |
| logit | 0.2688 | 0.5699 | 6 |
| tuned | 0.2796 | 0.5699 | 6 |
