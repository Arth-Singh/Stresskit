"""Shard one battery's finder calls across several processes (one per GPU).

Set ``STRESSKIT_SHARD=i/n`` in worker ``i`` of ``n``. Every worker runs the
same battery; a finder wrapped with :meth:`Shard.run` computes only the runs
whose key hashes into its shard, writes each finding to a shared cache, and
returns a placeholder for the others. A final process with the variable
unset finds every run in the cache and writes the card; it computes any run
a worker did not finish. Workers must not write artifacts
(:attr:`Shard.is_worker`).

The key is derived from the finder's own inputs (a digest of the data, the
seed and the config), so it is stable across processes and independent of
the battery's run order.
"""

import hashlib
import json
import os

import stresskit as sk


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Shard:
    def __init__(self, cache_dir):
        spec = os.environ.get("STRESSKIT_SHARD")
        self.index, self.count = (int(x) for x in spec.split("/")) if spec else (0, 1)
        if not 0 <= self.index < self.count:
            raise ValueError(f"STRESSKIT_SHARD={spec!r}: index out of range")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def is_worker(self):
        return "STRESSKIT_SHARD" in os.environ

    def key(self, data, seed, config):
        return digest([digest(data), seed, config])[:20]

    def mine(self, key):
        return int(key, 16) % self.count == self.index

    def _path(self, key):
        return os.path.join(self.cache_dir, f"finding_{key}.json")

    def load(self, key):
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        return sk.Finding(components=set(d["components"]) if d["components"] is not None else None,
                          universe_size=d.get("universe_size"), claim=d["claim"],
                          score=d["score"], meta=d["meta"])

    def save(self, key, finding):
        payload = {"components": (sorted(str(c) for c in finding.components)
                                  if finding.has_structure() else None),
                   "universe_size": finding.universe_size, "claim": finding.claim,
                   "score": finding.score, "meta": finding.meta}
        tmp = self._path(key) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1, default=str)
        os.replace(tmp, self._path(key))

    def run(self, compute, data, seed, config, placeholder):
        """Return the cached finding, compute it if this shard owns it, or
        return ``placeholder`` (a Finding) for another shard's run."""
        key = self.key(data, seed, config)
        cached = self.load(key)
        if cached is not None:
            return cached
        if self.is_worker and not self.mine(key):
            return placeholder
        finding = compute()
        self.save(key, finding)
        return finding
