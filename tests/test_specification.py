import pytest

from stresskit.metrics import mean_pairwise_jaccard
from stresskit.specification import SpecificationSpace


def test_crossed_manifest_has_normalized_product_mass():
    space = SpecificationSpace(
        axes={"seed_policy": ["fixed", "varied"], "threshold": [0.1, 0.2, 0.4]},
        weights={"threshold": [1, 2, 1]},
    )
    manifest = space.enumerate_manifest()
    assert space.size == 6
    assert len(manifest) == 6
    assert sum(row["target_probability"] for row in manifest) == pytest.approx(1.0)
    assert all(row["design"] == "crossed_enumeration" for row in manifest)


def test_sample_manifest_is_reproducible_and_records_probability():
    space = SpecificationSpace(
        axes={"a": [0, 1], "b": ["x", "y"]},
        weights={"a": [3, 1]},
    )
    first = space.sample_manifest(20, seed=17)
    second = space.sample_manifest(20, seed=17)
    assert first == second
    assert all(
        row["target_probability"] == space.probability(row["configuration"])
        for row in first
    )


def test_oat_can_hide_registered_axis_interaction():
    space = SpecificationSpace(axes={"a": [0, 1], "b": [0, 1]})
    stable = frozenset({1, 2, 3})
    interaction = frozenset({4, 5, 6})

    def finding(configuration):
        return interaction if configuration == {"a": 1, "b": 1} else stable

    oat = [
        finding(row["configuration"])
        for row in space.diagnostic_oat_manifest({"a": 0, "b": 0})
    ]
    crossed = [
        finding(row["configuration"])
        for row in space.enumerate_manifest()
    ]
    assert mean_pairwise_jaccard(oat) == 1.0
    assert mean_pairwise_jaccard(crossed) == 0.5


def test_invalid_space_and_configuration_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SpecificationSpace(axes={})
    with pytest.raises(ValueError, match="unknown axes"):
        SpecificationSpace(axes={"a": [1]}, weights={"b": [1]})
    space = SpecificationSpace(axes={"a": [1, 2]})
    with pytest.raises(ValueError, match="every axis"):
        space.probability({})
