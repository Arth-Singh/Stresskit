# Lens baseline comparison — Qwen/Qwen3.5-27B

Hit criterion: every intermediate of an item at rank <= 5 (word-like mask, all fitted layers, position -1). `found@100`: intermediate appears anywhere in the top-100.

## lens-eval-association

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.1569 | 0.4412 | 9 |
| logit | 0.1373 | 0.3529 | 8 |

## lens-eval-multihop

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.5054 | 0.6989 | 2 |
| logit | 0.3978 | 0.6989 | 5 |
