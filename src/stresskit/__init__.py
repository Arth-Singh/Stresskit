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

__version__ = "1.0.0.dev0"

from .finding import (
    Finding, circuit, direction, feature_set, probe, findings_from_jsonl,
)
from .battery import (
    stress, from_findings, from_jsonl, verdict_trace, verdict_trace_markdown,
    decision_state, confirmatory_verdict,
    Thresholds, StressResult, RunRecord, DEFAULT_BATTERY,
)
from .card import (
    StabilityCard, load_card, validate_card_dict, verify_card_dict,
    verify_oracle_report_dict, verify_artifact_dict, classify_artifact_dict,
)
from .compare import compare_cards, compare_markdown
from .report import generate_checklist
from .oracle import stress_oracle, OracleProbe, OracleThresholds, blind_spot_matrix
from .specification import SpecificationSpace
from .confirmatory import (
    ConfirmatoryCard, ConfirmatoryResult, confirmatory_from_findings,
    verify_confirmatory_card_dict,
)
from .utility import (
    Baseline, PredictionBaseline, UtilityMetricSpec, attach_utility,
    build_utility_evidence, utility_block, utility_check,
    verify_utility_evidence,
)
from .audit_models import (
    SourceBundle, ClaimRecord, AgentOpinion, AuditSpec, ResourcePlan,
    RunAttestation, AuditBundle, AuditDecision,
)
from .audit_compile import (
    compile_claim_record, detect_prompt_injection, discover_claims,
    freeze_audit_spec, make_resource_plan, regenerate_run_manifest,
)
from .audit_verify import verify_audit_bundle, verify_audit_release
from .audit_profiles import (
    PROFILE_REGISTRY, PROFILE_REGISTRY_DIGEST, get_profile,
    holm_bonferroni,
)
from .integrity import ContentAddressedStore, ContentRef
from . import metrics, baselines, judges, oracle, utility

__all__ = [
    "__version__",
    "Finding",
    "circuit",
    "direction",
    "feature_set",
    "probe",
    "stress",
    "from_findings",
    "from_jsonl",
    "findings_from_jsonl",
    "verdict_trace",
    "verdict_trace_markdown",
    "decision_state",
    "confirmatory_verdict",
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
    "Baseline",
    "PredictionBaseline",
    "UtilityMetricSpec",
    "utility_block",
    "utility_check",
    "attach_utility",
    "build_utility_evidence",
    "verify_utility_evidence",
    "stress_oracle",
    "OracleProbe",
    "OracleThresholds",
    "blind_spot_matrix",
    "SpecificationSpace",
    "ConfirmatoryCard",
    "ConfirmatoryResult",
    "confirmatory_from_findings",
    "verify_confirmatory_card_dict",
    "SourceBundle",
    "ClaimRecord",
    "AgentOpinion",
    "AuditSpec",
    "ResourcePlan",
    "RunAttestation",
    "AuditBundle",
    "AuditDecision",
    "ContentAddressedStore",
    "ContentRef",
    "compile_claim_record",
    "detect_prompt_injection",
    "discover_claims",
    "freeze_audit_spec",
    "make_resource_plan",
    "regenerate_run_manifest",
    "verify_audit_bundle",
    "verify_audit_release",
    "PROFILE_REGISTRY",
    "PROFILE_REGISTRY_DIGEST",
    "get_profile",
    "holm_bonferroni",
    "metrics",
    "baselines",
    "judges",
    "oracle",
    "utility",
]
