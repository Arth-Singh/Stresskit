"""Declared specification spaces for confirmatory StressKit audits.

Diagnostic one-at-a-time sweeps live in :mod:`stresskit.battery`. This module
constructs crossed or probability-sampled manifests with an explicit target
distribution, preventing pooled OAT runs from masquerading as confirmatory
samples.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SpecificationSpace:
    """Finite product specification space with per-axis probability weights."""

    axes: Mapping[str, Sequence[Any]]
    weights: Optional[Mapping[str, Sequence[float]]] = None

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("specification space requires at least one axis")
        for name, values in self.axes.items():
            if not name or not values:
                raise ValueError(f"axis {name!r} must have a name and values")
        unknown = set(self.weights or {}) - set(self.axes)
        if unknown:
            raise ValueError(f"weights supplied for unknown axes: {sorted(unknown)}")
        for name in self.axis_names:
            raw = list((self.weights or {}).get(name, ()))
            if raw and len(raw) != len(self.axes[name]):
                raise ValueError(
                    f"axis {name!r} has {len(self.axes[name])} values but "
                    f"{len(raw)} weights"
                )
            if raw and (
                any(not math.isfinite(value) or value < 0.0 for value in raw)
                or sum(raw) <= 0.0
            ):
                raise ValueError(f"axis {name!r} weights must be finite and nonnegative")

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.axes))

    @property
    def size(self) -> int:
        return math.prod(len(self.axes[name]) for name in self.axis_names)

    def axis_probabilities(self, name: str) -> List[float]:
        values = self.axes[name]
        raw = list((self.weights or {}).get(name, ()))
        if not raw:
            return [1.0 / len(values)] * len(values)
        total = sum(raw)
        return [value / total for value in raw]

    def probability(self, configuration: Mapping[str, Any]) -> float:
        """Product-distribution mass of one valid crossed configuration."""
        if set(configuration) != set(self.axis_names):
            raise ValueError("configuration must contain every axis exactly once")
        probability = 1.0
        for name in self.axis_names:
            values = list(self.axes[name])
            try:
                index = values.index(configuration[name])
            except ValueError as error:
                raise ValueError(
                    f"value {configuration[name]!r} is not registered for axis {name!r}"
                ) from error
            probability *= self.axis_probabilities(name)[index]
        return probability

    def enumerate_manifest(self) -> List[Dict[str, Any]]:
        """Enumerate full crossed space with exact target masses."""
        manifest = []
        value_lists = [self.axes[name] for name in self.axis_names]
        for draw_index, values in enumerate(itertools.product(*value_lists)):
            configuration = dict(zip(self.axis_names, values))
            manifest.append(
                {
                    "draw_index": draw_index,
                    "configuration": configuration,
                    "target_probability": self.probability(configuration),
                    "design": "crossed_enumeration",
                }
            )
        return manifest

    def sample_manifest(self, n_runs: int, seed: int) -> List[Dict[str, Any]]:
        """Draw IID configurations from declared product distribution."""
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")
        rng = random.Random(seed)
        manifest = []
        for draw_index in range(n_runs):
            configuration = {}
            for name in self.axis_names:
                configuration[name] = rng.choices(
                    list(self.axes[name]),
                    weights=self.axis_probabilities(name),
                    k=1,
                )[0]
            manifest.append(
                {
                    "draw_index": draw_index,
                    "configuration": configuration,
                    "target_probability": self.probability(configuration),
                    "design": "iid_specification_sample",
                    "manifest_seed": seed,
                }
            )
        return manifest

    def diagnostic_oat_manifest(
        self, base: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        """Construct labeled OAT variants for localization, never certification."""
        self.probability(base)
        rows = [
            {
                "axis": "base",
                "configuration": dict(base),
                "design": "diagnostic_oat",
            }
        ]
        for name in self.axis_names:
            for value in self.axes[name]:
                if value == base[name]:
                    continue
                configuration = dict(base)
                configuration[name] = value
                rows.append(
                    {
                        "axis": name,
                        "configuration": configuration,
                        "design": "diagnostic_oat",
                    }
                )
        return rows
