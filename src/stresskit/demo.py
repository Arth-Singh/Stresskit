"""``stresskit demo`` — the dead-salmon test, in 30 seconds, no GPU.

One toy discovery method (top-k correlation selection over 200 candidate
features), stressed twice with the identical battery:

1. on data with a real effect — the finding survives;
2. on pure noise, where **no effect exists** — the method still returns a
   confident-looking set of 8 "responsible features" with a claim attached,
   every single run.

The two outputs are indistinguishable by looking at them. Only the battery
tells them apart. That asymmetry — same method, same output format, same
confidence, different grade — is the entire pitch, so this is the first
thing a new user should run:

    pip install stress-kit && stresskit demo
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from . import battery as _battery
from .finding import feature_set

N_FEATURES = 200
TRUE = frozenset({3, 17, 42, 88, 105, 133, 150, 190})


def _make_data(n_examples: int, noise: float, seed: int,
               signal: float = 1.0) -> List[Tuple[List[float], float]]:
    rng = random.Random(seed)
    data = []
    for _ in range(n_examples):
        x = [rng.gauss(0, 1) for _ in range(N_FEATURES)]
        label = signal * sum(x[i] for i in TRUE) + rng.gauss(0, noise)
        data.append((x, label))
    return data


def _finder(data: Any, seed: int, config: Dict[str, Any]):
    """Toy discovery: rank features by |correlation| with the label on a
    seed-dependent 75% subsample, keep the top-k — the shape of any
    minibatched attribution method."""
    k = config.get("k", 8)
    rng = random.Random(seed)
    data = rng.sample(list(data), max(4, int(0.75 * len(data))))
    n = len(data)
    means = [sum(x[i] for x, _ in data) / n for i in range(N_FEATURES)]
    label_mean = sum(y for _, y in data) / n
    corrs = []
    for i in range(N_FEATURES):
        cov = sum((x[i] - means[i]) * (y - label_mean) for x, y in data)
        var_x = sum((x[i] - means[i]) ** 2 for x, _ in data) or 1e-9
        var_y = sum((y - label_mean) ** 2 for _, y in data) or 1e-9
        corrs.append(abs(cov) / (var_x * var_y) ** 0.5)
    top = sorted(range(N_FEATURES), key=lambda i: -corrs[i])[:k]
    claim = ("first-half" if sum(1 for i in top if i < 100) >= k / 2
             else "second-half")
    recovered = len(set(top) & TRUE) / len(TRUE)
    return feature_set(top, claim=claim, score=recovered,
                       universe_size=N_FEATURES)


def _stress(data: Any, tag: str) -> "_battery.StressResult":
    return _battery.stress(
        _finder, data,
        battery=["seeds", "bootstrap", "hyperparams"],
        n_runs=8,
        config={"k": 8},
        hyperparams={"k": [6, 12]},
        claim_statement=f"The behavior is driven by 8 specific features ({tag})",
        model="toy-linear-model", task="synthetic-attribution",
        method="top-k correlation selector",
    )


def run_demo(html_dir: Optional[str] = None, echo=print) -> Dict[str, Any]:
    """Run the two-battery demo; returns both results for programmatic use."""
    echo("StressKit demo — one discovery method, two datasets, no GPU.\n")

    echo("[1/2] REAL EFFECT: 8 features genuinely drive the label "
         "(400 examples, low noise)")
    real = _stress(_make_data(400, noise=0.5, seed=0), "real effect")
    echo(f"      → grade {real.grade}: "
         f"J={real.pooled['mean_pairwise_jaccard']:.2f}, "
         f"claim flips {real.pooled['flip_rate']:.0%}, "
         f"score CV {real.pooled['score_cv']:.2f}\n")

    echo("[2/2] PURE NOISE: the label is random — NO effect exists "
         "(100 examples). Same method.")
    null = _stress(_make_data(100, noise=1.0, seed=0, signal=0.0), "pure noise")
    echo(f"      → the finder still returned 8 confident 'responsible "
         f"features' with a claim, all {len(null.runs)} runs")
    echo(f"      → grade {null.grade}: "
         f"J={null.pooled['mean_pairwise_jaccard']:.2f}, "
         f"claim flips {null.pooled['flip_rate']:.0%}, "
         f"score CV {null.pooled['score_cv']:.2f}\n")

    echo("Side by side:\n")
    echo("  |                    | real effect | pure noise |")
    echo("  |--------------------|-------------|------------|")
    echo("  | looks like         | a finding   | a finding  |")
    echo(f"  | structural overlap | J={real.pooled['mean_pairwise_jaccard']:.2f}      "
         f"| J={null.pooled['mean_pairwise_jaccard']:.2f}     |")
    echo(f"  | grade              | {real.grade}           | {null.grade}          |")
    echo("")
    echo("Both runs produce plausible components, a qualitative claim, and a")
    echo("score. Nothing about the null output looks wrong — only re-running")
    echo("under perturbation exposes it. If your pipeline has never been run")
    echo("through a battery, you do not know which column you are in.")
    echo("")
    echo("Next steps:")
    echo("  wrap your own finder      → https://github.com/Arth-Singh/Stresskit#quickstart")
    echo("  grade runs you have       → sk.from_jsonl('sweep.jsonl')")
    echo("  published findings graded → SCOREBOARD.md in the repo")

    if html_dir:
        from .htmlcard import card_html

        os.makedirs(html_dir, exist_ok=True)
        for slug, res in (("real_effect", real), ("pure_noise", null)):
            path = os.path.join(html_dir, f"demo_{slug}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(card_html(res.card.to_dict()))
            echo(f"  HTML card written        → {path}")

    return {"real": real, "null": null}
