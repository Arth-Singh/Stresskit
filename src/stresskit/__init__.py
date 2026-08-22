"""StressKit — the stability and sanity-check harness for mechanistic
interpretability claims.

    import stresskit as sk

    def my_finder(data, seed, config) -> sk.Finding:
        edges = discover_circuit(data, seed=seed, **config)
        return sk.circuit(edges, claim=band_of(edges), score=faithfulness(edges),
                          universe_size=N_EDGES)

    result = sk.stress(my_finder, data,
                       battery=["seeds", "bootstrap", "hyperparams"],
                       hyperparams={"threshold": [0.05, 0.2]})
    print(result.to_markdown())
    result.card.save("stability_card.json")
"""

__version__ = "0.3.0"

from .finding import Finding, circuit, feature_set, probe, findings_from_jsonl
from .battery import (
    stress, from_findings, from_jsonl, verdict_trace, verdict_trace_markdown,
    Thresholds, StressResult, RunRecord, DEFAULT_BATTERY,
)
from .card import (
    StabilityCard, load_card, validate_card_dict, verify_card_dict,
    verify_oracle_report_dict, verify_artifact_dict, classify_artifact_dict,
)
from .compare import compare_cards, compare_markdown
from .report import generate_checklist
from .oracle import stress_oracle, OracleProbe, OracleThresholds, blind_spot_matrix
from . import metrics, baselines, judges, oracle

__all__ = [
    "__version__",
    "Finding",
    "circuit",
    "feature_set",
    "probe",
    "stress",
    "from_findings",
    "from_jsonl",
    "findings_from_jsonl",
    "verdict_trace",
    "verdict_trace_markdown",
    "Thresholds",
    "StressResult",
    "RunRecord",
    "DEFAULT_BATTERY",
    "StabilityCard",
    "load_card",
    "validate_card_dict",
    "verify_card_dict",
    "verify_oracle_report_dict",
    "verify_artifact_dict",
    "classify_artifact_dict",
    "compare_cards",
    "compare_markdown",
    "generate_checklist",
    "stress_oracle",
    "OracleProbe",
    "OracleThresholds",
    "blind_spot_matrix",
    "metrics",
    "baselines",
    "judges",
    "oracle",
]
