"""SAE stability audit example — seed consistency + redundancy, CPU-only.

We fake three "SAE training runs" over the same synthetic feature dictionary:
each run recovers most true directions (permuted, sign-flipped, slightly
noisy) but also learns some run-specific junk and some duplicated features.
The audit quantifies both failure modes.

With a real SAE, replace the synthetic decoders with e.g.:

    sae_a = SAE.load_from_pretrained(...)      # SAELens
    W = sae_a.W_dec.detach().cpu().numpy()     # (n_features, d_model)

Run:  python examples/sae_audit.py   (requires numpy; scipy recommended)
"""

import numpy as np

from stresskit.adapters import sae

rng = np.random.default_rng(0)

D_MODEL, N_TRUE, N_LATENTS = 128, 96, 128
TRUE_DIRS = rng.normal(size=(N_TRUE, D_MODEL))


def fake_sae_run(seed, recovery=0.8, dup_features=6):
    """Simulate one SAE training run's decoder."""
    r = np.random.default_rng(seed)
    n_recovered = int(N_TRUE * recovery)
    picked = r.choice(N_TRUE, size=n_recovered, replace=False)
    recovered = TRUE_DIRS[picked] + r.normal(scale=0.05, size=(n_recovered, D_MODEL))
    signs = r.choice([-1.0, 1.0], size=(n_recovered, 1))
    recovered *= signs
    junk = r.normal(size=(N_LATENTS - n_recovered - dup_features, D_MODEL))
    dups = recovered[r.choice(n_recovered, size=dup_features)] * r.uniform(0.9, 1.1)
    W = np.concatenate([recovered, junk, dups])
    return W[r.permutation(len(W))]


decoders = [fake_sae_run(s) for s in (1, 2, 3)]

print("=" * 60)
print("1. Seed consistency (MCC across runs)")
print("=" * 60)
consistency = sae.seed_consistency(decoders)
print(f"mean MCC : {consistency['mean_mcc']:.3f}")
print(f"min MCC  : {consistency['min_mcc']:.3f}")
print(f"pairwise : {consistency['pairwise_mcc']}")
print()
print("Interpretation: MCC = 1.0 would mean every feature replicates across")
print("seeds up to permutation/sign. Anything well below 1.0 means part of")
print("your feature dictionary is a seed artifact — report it.")

print()
print("=" * 60)
print("2. Redundancy audit (near-duplicate features within one run)")
print("=" * 60)
audit = sae.redundancy_audit(decoders[0], threshold=0.9)
print(f"features            : {audit['n_features']}")
print(f"duplicate pairs     : {audit['n_duplicate_pairs']}")
print(f"redundant features  : {audit['n_redundant_features']} "
      f"({audit['redundant_fraction']:.1%})")
print(f"duplicate clusters  : {audit['n_clusters']} "
      f"(largest: {audit['largest_cluster']})")
