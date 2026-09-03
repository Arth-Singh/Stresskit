import hashlib
import json
from pathlib import Path

from stresskit.battery_calibration import battery_calibration_source_digest
from stresskit.calibration import calibration_source_digest
from stresskit.extended_validation import extended_validation_source_digest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "calibration"

# every frozen study, and the digest of the sources its result depends on
SOURCE_DIGEST = {
    "structural_known_truth_pilot": calibration_source_digest,
    "extended_validation": extended_validation_source_digest,
    "battery_known_truth": battery_calibration_source_digest,
}
# a study whose family name is the artifact's filename prefix
FAMILY_SEEDS = {
    "S1-S5": {20260824, 20260825},
    "S6-S9": {20260824, 20260825},
    "battery-known-truth": {20260904, 20260905},
}


def manifest():
    return json.loads((ARTIFACTS / "manifest.json").read_text())


def family_of(path):
    for family in FAMILY_SEEDS:
        if path.startswith(family):
            return family
    raise AssertionError(f"frozen artifact {path!r} belongs to no known study family")


def test_frozen_calibration_manifest_hashes_and_sources_match():
    rows = manifest()["artifacts"]
    assert manifest()["status"] == "frozen_method_validation"
    assert rows, "the manifest lists no frozen artifact"
    for row in rows:
        path = ARTIFACTS / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        payload = json.loads(path.read_text())
        assert payload["provenance"]["source_sha256"] == row["source_sha256"]
        study = payload["study"]
        # S6--S9 does not name itself in the payload
        digest = SOURCE_DIGEST.get(study, extended_validation_source_digest)
        assert row["source_sha256"] == digest()


def test_frozen_studies_have_primary_and_disjoint_seed_replication():
    grouped = {}
    for row in manifest()["artifacts"]:
        grouped.setdefault(family_of(row["path"]), []).append(row)
    assert set(grouped) == set(FAMILY_SEEDS), (
        "every frozen study family must be represented: "
        f"{sorted(grouped)} against {sorted(FAMILY_SEEDS)}"
    )
    for family, rows in grouped.items():
        assert {row["role"] for row in rows} == {
            "primary",
            "disjoint_seed_replication",
        }, family
        assert {row["master_seed"] for row in rows} == FAMILY_SEEDS[family], family
