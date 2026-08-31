# Lens baseline comparison — Qwen/Qwen3.6-27B

Hit criterion: every intermediate of an item at rank <= 5 (word-like mask, all fitted layers, position -1). `found@100`: intermediate appears anywhere in the top-100.

## lens-eval-association

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.1373 | 0.3431 | 8 |
| logit | 0.1373 | 0.2941 | 12 |

## lens-eval-multihop

| lens | hit@5 | found@100 | median rank when found |
|---|---|---|---|
| jlens | 0.4946 | 0.7097 | 3 |
| logit | 0.3763 | 0.6882 | 5 |
