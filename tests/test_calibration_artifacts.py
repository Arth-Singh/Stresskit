import hashlib
import json
from pathlib import Path

from stresskit.calibration import calibration_source_digest
from stresskit.extended_validation import extended_validation_source_digest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "calibration"


def test_frozen_calibration_manifest_hashes_and_sources_match():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    assert manifest["status"] == "frozen_method_validation"
    assert len(manifest["artifacts"]) == 4
    for row in manifest["artifacts"]:
        path = ARTIFACTS / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        payload = json.loads(path.read_text())
        assert payload["provenance"]["source_sha256"] == row["source_sha256"]
        expected_source = (
            calibration_source_digest()
            if payload["study"] == "structural_known_truth_pilot"
            else extended_validation_source_digest()
        )
        assert row["source_sha256"] == expected_source


def test_frozen_studies_have_primary_and_disjoint_seed_replication():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    grouped = {}
    for row in manifest["artifacts"]:
        family = "S1-S5" if row["path"].startswith("S1-S5") else "S6-S9"
        grouped.setdefault(family, []).append(row)
    for rows in grouped.values():
        assert {row["role"] for row in rows} == {
            "primary", "disjoint_seed_replication"
        }
        assert {row["master_seed"] for row in rows} == {20260824, 20260825}
