"""Nested ``stresskit audit`` CLI implementing v1 artifact lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .audit_compile import (
    compile_claim_record,
    discover_claims,
    freeze_audit_spec,
    make_resource_plan,
)
from .audit_models import (
    AgentOpinion,
    AuditBundle,
    AuditDecision,
    AuditSpec,
    ClaimRecord,
    ResourcePlan,
    SourceBundle,
)
from .audit_verify import verify_audit_bundle, verify_audit_release
from .audit_worker import (
    ExecutorCapabilities,
    assemble_audit_bundle,
    attest_failure,
    attest_success,
    compatible_executor,
)
from .agent_runner import (
    OpenRouterClient,
    OpenRouterRouteBinding,
    generate_openrouter_opinion,
)
from .evidence import PublicationBlocked, build_evidence_board, write_evidence_site
from .integrity import ContentAddressedStore, ContentRef, read_secret
from .source_intake import build_source_bundle
from .source_extract import write_extracted_source


def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write(value: Any, path: Optional[str]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)


def _read_utf8_bytes(path: Any) -> str:
    """Decode exact UTF-8 bytes without universal-newline translation."""
    return Path(path).read_bytes().decode("utf-8")


def _source_texts(values: Sequence[str]) -> Dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source-text must be DOCUMENT_ID=PATH")
        document_id, path = value.split("=", 1)
        if not document_id or not path:
            raise ValueError("--source-text must be DOCUMENT_ID=PATH")
        output[document_id] = _read_utf8_bytes(path)
    return output


def _closure_inputs(values: Sequence[str]) -> Dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--closure-input must be SHA256_DIGEST=PATH")
        digest, path = value.split("=", 1)
        if not digest or not path:
            raise ValueError("--closure-input must be SHA256_DIGEST=PATH")
        if digest in output:
            raise ValueError(f"duplicate --closure-input digest: {digest}")
        output[digest] = path
    return output


def _opinions(paths: Sequence[str]) -> List[AgentOpinion]:
    return [AgentOpinion.from_dict(_load(path)) for path in paths]


def _cmd_source(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    bundle, references = build_source_bundle(
        _load(args.manifest), ContentAddressedStore(args.cas),
        base_dir=manifest_path.parent,
        closure_inputs=_closure_inputs(getattr(args, "closure_input", ())),
    )
    _write(bundle.to_dict(), args.output)
    _write([reference.to_dict() for reference in references], args.closure_output)
    return 0


def _cmd_extract_source(args: argparse.Namespace) -> int:
    write_extracted_source(args.input, args.output)
    return 0


def _cmd_opinion(args: argparse.Namespace) -> int:
    source = SourceBundle.from_dict(_load(args.source))
    texts = _source_texts(args.source_text)
    extractors = _opinions(args.extractor_opinion)
    panel_path = Path(args.panel_plan)
    binding = OpenRouterRouteBinding.from_panel_plan(
        _load(args.panel_plan), opinion_id=args.opinion_id
    )
    claim_query = _read_utf8_bytes(panel_path.parent / binding.claim_query)
    opinion, references = generate_openrouter_opinion(
        source,
        texts,
        binding=binding,
        claim_query=claim_query,
        store=ContentAddressedStore(args.cas),
        client=OpenRouterClient.from_environment(),
        extractor_opinions=extractors,
        timeout=args.timeout,
    )
    _write(opinion.to_dict(), args.output)
    _write([reference.to_dict() for reference in references], args.closure_output)
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    source = SourceBundle.from_dict(_load(args.source))
    texts = _source_texts(args.source_text)
    result = discover_claims(
        source, _opinions(args.opinions), source_texts=texts or None
    )
    _write(result, args.output)
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    source = SourceBundle.from_dict(_load(args.source))
    texts = _source_texts(args.source_text)
    result = compile_claim_record(
        source, _opinions(args.opinions), _load(args.template),
        source_texts=texts or None,
    )
    _write(result, args.output)
    return 0


def _claim_from_payload(payload: Mapping[str, Any]) -> ClaimRecord:
    if payload.get("artifact") == ClaimRecord.ARTIFACT:
        return ClaimRecord.from_dict(payload)
    if payload.get("artifact") == "stresskit_compilation_result" and \
            isinstance(payload.get("claim_record"), Mapping):
        return ClaimRecord.from_dict(payload["claim_record"])
    raise ValueError("freeze input must be ClaimRecord or successful compilation result")


def _cmd_freeze(args: argparse.Namespace) -> int:
    claim = _claim_from_payload(_load(args.claim))
    design_file = _load(args.design)
    if not isinstance(design_file, dict):
        raise ValueError("audit design file must be a JSON object")
    audit_id = str(design_file.pop("audit_id"))
    frozen_at = design_file.pop("frozen_at", None)
    multiplicity = dict(design_file.pop("multiplicity_family"))
    reproducibility = dict(design_file.pop("reproducibility"))
    external_validation = str(design_file.pop(
        "external_validation", "not obtained"
    ))
    spec = freeze_audit_spec(
        claim, design_file, audit_id=audit_id, frozen_at=frozen_at,
        multiplicity_family=multiplicity,
        reproducibility=reproducibility,
        external_validation=external_validation,
    )
    _write(spec.to_dict(), args.output)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    spec = AuditSpec.from_dict(_load(args.spec))
    plan = make_resource_plan(
        spec, _load(args.resources), key=read_secret(args.signing_key_file),
        key_id=args.key_id, signing_algorithm=args.signing_algorithm,
        created_at=args.created_at,
    )
    _write(plan.to_dict(), args.output)
    if args.output:
        print(
            f"ResourcePlan written to {args.output}. Audit paused; supply compatible "
            "outbound-polling executor."
        )
    return 0


def _capabilities(payload: Mapping[str, Any]) -> ExecutorCapabilities:
    isolation = payload.get("isolation", {})
    return ExecutorCapabilities(
        executor_id=str(payload.get("executor_id", "")),
        hardware_classes=list(payload.get("hardware_classes", [])),
        gpu_count=int(payload.get("gpu_count", 0)),
        cpu_count=int(payload.get("cpu_count", 0)),
        storage_bytes=int(payload.get("storage_bytes", 0)),
        network_namespace=isolation.get("network_namespace") is True,
        credential_isolation=isolation.get("credential_isolation") is True,
        read_only_mounts=isolation.get("read_only_mounts") is True,
        scratch_quota=isolation.get("scratch_quota") is True,
        output_allowlist=isolation.get("output_allowlist") is True,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    spec = AuditSpec.from_dict(_load(args.spec))
    plan = ResourcePlan.from_dict(_load(args.plan))
    capabilities = _capabilities(_load(args.capabilities))
    compatible, problems = compatible_executor(plan, capabilities)
    if not compatible:
        raise ValueError("executor incompatible with ResourcePlan: " + "; ".join(problems))
    store = ContentAddressedStore(args.cas)
    key = read_secret(args.signing_key_file)
    run_dir = Path(args.run_dir)
    references = [ContentRef.from_dict(row) for row in _load(args.closure)]
    attestations = []
    for slot in spec.run_manifest:
        slot_id = str(slot["slot_id"])
        output_path = run_dir / f"{slot_id}.json"
        status_path = run_dir / f"{slot_id}.status.json"
        environment_id = f"{args.execution_prefix}-{slot['cohort']}-{slot_id}"
        if output_path.is_file():
            raw = _load(str(output_path))
            if not isinstance(raw, Mapping):
                raise ValueError(f"{output_path} must contain JSON object")
            attestation, reference = attest_success(
                spec, plan, slot, raw, store,
                executor_id=capabilities.executor_id,
                execution_environment_id=environment_id,
                key=key, key_id=args.key_id,
                signing_algorithm=args.signing_algorithm,
            )
        elif status_path.is_file():
            status_payload = _load(str(status_path))
            status = str(status_payload.get("status", "failed"))
            error = status_payload.get("error", status_payload)
            if not isinstance(error, Mapping):
                error = {"message": str(error)}
            attestation, reference = attest_failure(
                spec, plan, slot, status, error, store,
                executor_id=capabilities.executor_id,
                execution_environment_id=environment_id,
                key=key, key_id=args.key_id,
                signing_algorithm=args.signing_algorithm,
            )
        else:
            attestation, reference = attest_failure(
                spec, plan, slot, "missing",
                {"message": "executor returned no output or terminal status"}, store,
                executor_id=capabilities.executor_id,
                execution_environment_id=environment_id,
                key=key, key_id=args.key_id,
                signing_algorithm=args.signing_algorithm,
            )
        attestations.append(attestation)
        references.append(reference)
    bundle = assemble_audit_bundle(spec, plan, attestations, references)
    _write(bundle.to_dict(), args.output)
    return 0


def _trusted_keys(values: Sequence[str]) -> Dict[str, bytes]:
    keys = {}
    for value in values:
        if "=" not in value:
            raise ValueError("trusted key must be KEY_ID=PATH")
        key_id, path = value.split("=", 1)
        if key_id in keys:
            raise ValueError(f"duplicate trusted key ID {key_id!r}")
        keys[key_id] = read_secret(path)
    if not keys:
        raise ValueError("at least one trusted key is required per trust domain")
    return keys


def _json_report(value: Any) -> Any:
    if isinstance(value, AuditDecision):
        return value.to_dict()
    if isinstance(value, AuditSpec):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _json_report(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_report(child) for child in value]
    return value


def _cmd_verify(args: argparse.Namespace) -> int:
    bundles = [AuditBundle.from_dict(_load(path)) for path in args.bundles]
    store = ContentAddressedStore(args.cas)
    plan_keys = _trusted_keys(args.trusted_plan_key)
    executor_keys = _trusted_keys(args.trusted_executor_key)
    if len(bundles) == 1:
        report = verify_audit_bundle(
            bundles[0],
            store,
            trusted_plan_keys=plan_keys,
            trusted_executor_keys=executor_keys,
        )
        exit_code = 0 if report.get("verified") else 1
    else:
        report = verify_audit_release(
            bundles,
            store,
            trusted_plan_keys=plan_keys,
            trusted_executor_keys=executor_keys,
        )
        exit_code = 0 if report.get("ok") else 1
    _write(_json_report(report), args.output)
    return exit_code


def _cmd_publish(args: argparse.Namespace) -> int:
    registry = _load(args.registry)
    bundles = [AuditBundle.from_dict(_load(path)) for path in args.bundles]
    try:
        board = build_evidence_board(
            registry, bundles, ContentAddressedStore(args.cas),
            trusted_plan_keys=_trusted_keys(args.trusted_plan_key),
            trusted_executor_keys=_trusted_keys(args.trusted_executor_key),
            as_of=args.as_of,
            agent_only_review=args.agent_only_review,
        )
        write_evidence_site(board, args.output_dir)
    except PublicationBlocked as exc:
        print(f"publication blocked: {exc}", file=sys.stderr)
        return 2
    print(f"Verified evidence board written to {args.output_dir}")
    return 0


def add_audit_parser(subparsers: Any) -> None:
    """Register all ``stresskit audit`` lifecycle subcommands."""
    audit = subparsers.add_parser(
        "audit", help="autonomous claim-audit lifecycle and offline verification"
    )
    commands = audit.add_subparsers(dest="audit_command", required=True)

    source = commands.add_parser(
        "source", help="build content-addressed SourceBundle from local files"
    )
    source.add_argument("manifest")
    source.add_argument("--cas", required=True)
    source.add_argument(
        "--closure-input", action="append", default=[],
        metavar="SHA256_DIGEST=PATH",
        help=(
            "offline bytes for a manifest digest link; relative paths resolve "
            "from the manifest directory"
        ),
    )
    source.add_argument("--closure-output", required=True)
    source.add_argument("-o", "--output", required=True)
    source.set_defaults(func=_cmd_source)

    extract_source = commands.add_parser(
        "extract-source",
        help="deterministically extract UTF-8 text or notebook cell sources",
    )
    extract_source.add_argument("input")
    extract_source.add_argument("-o", "--output", required=True)
    extract_source.set_defaults(func=_cmd_extract_source)

    opinion = commands.add_parser(
        "opinion",
        help="generate one pinned, content-addressed AgentOpinion through OpenRouter",
    )
    opinion.add_argument("source")
    opinion.add_argument(
        "--panel-plan", required=True,
        help="frozen panel JSON binding model, provider, parameters, and claim query",
    )
    opinion.add_argument(
        "--opinion-id", required=True,
        help="select exactly one request row from --panel-plan",
    )
    opinion.add_argument(
        "--extractor-opinion", action="append", default=[], metavar="PATH",
        help="critic only: supply exactly two extractor AgentOpinion files",
    )
    opinion.add_argument(
        "--source-text", action="append", default=[], required=True,
        metavar="DOCUMENT_ID=PATH",
    )
    opinion.add_argument("--timeout", type=float, default=120.0)
    opinion.add_argument("--cas", required=True)
    opinion.add_argument("--closure-output", required=True)
    opinion.add_argument("-o", "--output", required=True)
    opinion.set_defaults(func=_cmd_opinion)

    discover = commands.add_parser("discover", help="combine isolated extractor and critic opinions")
    discover.add_argument("source")
    discover.add_argument("--opinions", nargs=3, required=True)
    discover.add_argument("--source-text", action="append", default=[], metavar="DOCUMENT_ID=PATH")
    discover.add_argument("-o", "--output")
    discover.set_defaults(func=_cmd_discover)

    compile_parser = commands.add_parser("compile", help="compile supported candidate into ClaimRecord")
    compile_parser.add_argument("source")
    compile_parser.add_argument("template")
    compile_parser.add_argument("--opinions", nargs=3, required=True)
    compile_parser.add_argument("--source-text", action="append", default=[], metavar="DOCUMENT_ID=PATH")
    compile_parser.add_argument("-o", "--output")
    compile_parser.set_defaults(func=_cmd_compile)

    freeze = commands.add_parser("freeze", help="freeze joint design and regenerate complete manifest")
    freeze.add_argument("claim")
    freeze.add_argument("design")
    freeze.add_argument("-o", "--output")
    freeze.set_defaults(func=_cmd_freeze)

    plan = commands.add_parser("plan", help="emit signed GPU/count/time/storage ResourcePlan")
    plan.add_argument("spec")
    plan.add_argument("resources")
    plan.add_argument("--signing-key-file", required=True)
    plan.add_argument("--key-id", required=True)
    plan.add_argument(
        "--signing-algorithm", choices=("ed25519", "hmac-sha256"),
        default="ed25519",
    )
    plan.add_argument("--created-at")
    plan.add_argument("-o", "--output")
    plan.set_defaults(func=_cmd_plan)

    run = commands.add_parser("run", help="ingest isolated executor outputs and attest every slot")
    run.add_argument("spec")
    run.add_argument("plan")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--cas", required=True)
    run.add_argument("--closure", required=True)
    run.add_argument("--capabilities", required=True)
    run.add_argument("--signing-key-file", required=True)
    run.add_argument("--key-id", required=True)
    run.add_argument(
        "--signing-algorithm", choices=("ed25519", "hmac-sha256"),
        default="ed25519",
    )
    run.add_argument("--execution-prefix", required=True)
    run.add_argument("-o", "--output")
    run.set_defaults(func=_cmd_run)

    verify = commands.add_parser("verify", help="recompute reducers, metrics, checks, and decision from raw closure")
    verify.add_argument("bundles", nargs="+")
    verify.add_argument("--cas", required=True)
    verify.add_argument(
        "--trusted-plan-key", action="append", default=[], metavar="KEY_ID=PATH"
    )
    verify.add_argument(
        "--trusted-executor-key", action="append", default=[], metavar="KEY_ID=PATH"
    )
    verify.add_argument("-o", "--output")
    verify.set_defaults(func=_cmd_verify)

    publish = commands.add_parser("publish", help="verify release and render complete evidence matrix")
    publish.add_argument("registry")
    publish.add_argument("bundles", nargs="*")
    publish.add_argument("--cas", required=True)
    publish.add_argument(
        "--trusted-plan-key", action="append", default=[], metavar="KEY_ID=PATH"
    )
    publish.add_argument(
        "--trusted-executor-key", action="append", default=[], metavar="KEY_ID=PATH"
    )
    publish.add_argument("--output-dir", required=True)
    publish.add_argument("--as-of")
    publish.add_argument("--agent-only-review", action="store_true")
    publish.set_defaults(func=_cmd_publish)
