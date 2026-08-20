"""StressKit quickstart — no GPU, no model downloads, runs in seconds.

We simulate the situation every interp researcher is in: a "discovery
method" that selects the components (here: feature indices) responsible for
a behavior, plus a qualitative claim about where they live. We stress-test
it twice:

1. a well-powered setting  -> the finding is real and survives the battery
2. an under-powered setting -> the same method produces a plausible-looking
   finding that falls apart under bootstrap + seed variation

Run:  python examples/quickstart_toy.py
"""

import os
import random

import stresskit as sk

N_FEATURES = 200                       # universe of candidate components
TRUE = frozenset({3, 17, 42, 88, 105, 133, 150, 190})  # ground truth


def make_data(n_examples, noise, seed, signal=1.0):
    """Each example: (values per feature, label). Label depends on TRUE feats
    scaled by ``signal`` (signal=0.0 -> the effect does not exist at all)."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_examples):
        x = [rng.gauss(0, 1) for _ in range(N_FEATURES)]
        label = signal * sum(x[i] for i in TRUE) + rng.gauss(0, noise)
        data.append((x, label))
    return data


def finder(data, seed, config):
    """A toy discovery method: rank features by |correlation| with the label
    on a seed-dependent 75% subsample (like any minibatched/subsampled real
    method), keep the top-k. Claim = which half of the 'model' it lives in."""
    k = config.get("k", 8)
    rng = random.Random(seed)
    data = rng.sample(list(data), max(4, int(0.75 * len(data))))
    n = len(data)
    # feature-label correlation (no numpy needed for the toy)
    means = [sum(x[i] for x, _ in data) / n for i in range(N_FEATURES)]
    label_mean = sum(y for _, y in data) / n
    corrs = []
    for i in range(N_FEATURES):
        cov = sum((x[i] - means[i]) * (y - label_mean) for x, y in data)
        var_x = sum((x[i] - means[i]) ** 2 for x, _ in data) or 1e-9
        var_y = sum((y - label_mean) ** 2 for _, y in data) or 1e-9
        corrs.append(abs(cov) / (var_x * var_y) ** 0.5)
    top = sorted(range(N_FEATURES), key=lambda i: -corrs[i])[:k]

    claim = "first-half" if sum(1 for i in top if i < 100) >= k / 2 else "second-half"
    recovered = len(set(top) & TRUE) / len(TRUE)  # score: ground-truth recovery
    return sk.feature_set(top, claim=claim, score=recovered,
                          universe_size=N_FEATURES)


def run(tag, n_examples, noise, signal=1.0):
    print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}")
    data = make_data(n_examples, noise, seed=0, signal=signal)
    result = sk.stress(
        finder,
        data,
        battery=["seeds", "bootstrap", "hyperparams"],
        n_runs=8,
        config={"k": 8},
        hyperparams={"k": [6, 12]},
        claim_statement="The behavior is driven by 8 specific features",
        model="toy-linear-model",
        task="synthetic-attribution",
        method="top-k correlation selector",
    )
    print(result)
    print()
    print(result.to_markdown())

    os.makedirs("examples/output", exist_ok=True)
    slug = tag.split()[0].lower()
    card_path = f"examples/output/card_{slug}.json"
    result.card.save(card_path)
    print(f"\n-> card saved to {card_path}")
    return result


if __name__ == "__main__":
    # Well-powered: lots of data, low noise. Expect grade A.
    good = run("WELL-POWERED setting (expect A)", n_examples=400, noise=0.5)

    # Null setting: the effect does not exist (signal=0), but the finder still
    # returns a confident-looking set of 8 "responsible features" with a claim
    # attached, every single time. Only the battery reveals it's noise.
    bad = run("NULL setting — no real effect (expect C/D)",
              n_examples=100, noise=1.0, signal=0.0)

    print("\nSummary:")
    print(f"  well-powered  -> grade {good.grade}, "
          f"J={good.pooled['mean_pairwise_jaccard']:.2f}, "
          f"flip={good.pooled['flip_rate']:.2f}")
    print(f"  null effect   -> grade {bad.grade}, "
          f"J={bad.pooled['mean_pairwise_jaccard']:.2f}, "
          f"flip={bad.pooled['flip_rate']:.2f}")
    print("\nSame method, same output format, same confidence — only the "
          "battery tells them apart. That is the point of StressKit.")
