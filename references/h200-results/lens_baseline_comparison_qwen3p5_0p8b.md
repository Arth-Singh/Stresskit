# Lens baseline comparison — Qwen/Qwen3.5-0.8B

Hit criterion: every intermediate of an item at rank <= 5 (word-like mask, all fitted layers, position -1). `found@100`: intermediate appears anywhere in the top-100.

## lens-eval-association

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.0196 | 0.098 | 29 |
| logit | 0.0 | 0.0392 | 51 |

## lens-eval-multihop

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.2581 | 0.5269 | 7 |
| logit | 0.2151 | 0.4839 | 7 |
