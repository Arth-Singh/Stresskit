# StressKit v1 evidence board

`stresskit audit publish` builds a static claim-level evidence matrix from a
frozen release registry and complete `AuditBundle`s.

Publisher performs release-level offline verification, recomputes global
Holm–Bonferroni decisions, enforces author-response and validation gates, and
then emits:

- `evidence-board.json`;
- `README.md`;
- `index.html`;
- one paper page containing only its claim rows.

Rows expose reproduction, stability/specificity, utility, generalization,
evidence confidence, and final status separately. They preserve registry order
and include exclusions and abstentions. No grade sorting, paper scalar, or
whole-paper verdict is implemented.

Example:

```bash
stresskit audit publish benchmark/release-v1.json bundles/*.json \
  --cas ./audit-cas \
  --trusted-plan-key control=keys/control-public.pem \
  --trusted-executor-key worker=keys/worker-public.pem \
  --output-dir ./public-evidence \
  --agent-only-review
```

`--agent-only-review` records an explicit choice. It does not change
`external_validation: not obtained`.

Named adverse rows publish only after 14 days from timezone-qualified
`notified_at`, or when `response_received: true` carries non-empty inline
`response_text`. Board copies complete response object and records exact
publication `as_of` timestamp.

Publisher requires new or empty output directory, preventing stale pages from
older release surviving beside newly verified evidence. Paper filenames include
claim-independent digest suffixes, preventing sanitized-ID collisions.
