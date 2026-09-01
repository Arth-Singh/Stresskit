# Mechanistic Tomography finite-effect-map intake

This directory freezes outcome-blind source, code-map, artifact, and claim-query
inputs for `mechtomo_finite_effect_map_recovery`. No model opinion, StressKit
experiment, upstream experiment, or confirmatory outcome was run while creating
this intake.

## Frozen sources

- Repository: `kwisatzh/mechanistic-tomography`
- Commit: `5c097d2175c632a4a15e359cb4d94ec923168472`
- Tree: `261ed565077cb55dc61cc004fd1bd7ef9be01c67`
- Repository archive: `sha256:9c0955019722da7b3b71cc9f2571c0f5310d23dd398b886c8f82c1aae4f0f5ec`
- Paper: `arXiv:2608.19338v1`
- arXiv PDF: `sha256:b3ff771db9b791691de6c3985833ca9ba3f3fc823214b5d57aad4c68f9df05c5`
- arXiv source archive: `sha256:a156f610419ef63c43d143ebb41d70ac1a97f10e0581d9b3f9f73cf27a4e61a9`
- Software license: Apache-2.0, pinned `LICENSE` digest
  `sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
- Paper license: CC-BY-4.0, pinned `LICENSE-PAPER.md` digest
  `sha256:48e4dc03652e5c1bc771ded0d56b3bcd182b11440566a3a93c1566a0fdcd2391`

`artifact-inventory.json` binds the released checkpoint, HMM configuration,
measurement matrix, aggregate measurements, direct coordinate-patching map,
upstream frozen output table, code paths, dependency ranges, and license review.
These are upstream evidence, not an independent StressKit reproduction.

## Frozen wording and corrected locator

`claim-query.txt` uses the registry `statement_to_extract` byte-for-byte:

> Orthogonal matching pursuit recovers the 32-coordinate finite-effect map with
> Pearson r = 0.989 and held-out R-squared = 0.935.

The exact support is in arXiv v1 Section 5.1, the paragraph beginning “After 12
aggregate measurements,” and Figure 2, PDF page 8. In primary TeX, it is lines
361–377. Table 2 is a different Section 5.2 finite-scale attribution-patching
calibration experiment. The candidate registry now names Section 5.1 and Figure
2 only. Frozen statement bytes remain unchanged.

The registry code map now uses `nt_mi_correspondence.py` for measurement
generation followed by `sparse_tomography_posthoc.py` for the OMP split,
selection, and metrics. The former `attribution_vs_finite_step0.py` path belongs
to the distinct Section 5.2 calibration study. `artifact-inventory.json` binds
the corrected path.

The released upstream row at `n_train=12` is OMP with 32 components, validation-
selected support size 4, `pearson_vs_mi=0.9886063036465091`, and
`holdout_r2=0.9347141098561826`; these values round to the paper sentence.

## Released HMM and dependencies

The checkpoint is tracked directly at `experiments/hmm/frozen/model.pt`, size
1,833,033 bytes, digest
`sha256:4d8689f8615cd2e78972f46dc022ae6b11d02eb2a4430bbae6cc013b0f299983`.
`config.json` freezes the synthetic HMM and model settings. No external dataset
is used. Intake only hashed and git-identified `model.pt`; it was not
deserialized. Any future PyTorch load must happen inside the declared isolated
executor because pickle-compatible checkpoints are executable input.

The full measurement generator requires Python 3.10 or 3.11 plus the upstream
ranges `torch>=2.3`, `numpy>=1.24`, `matplotlib>=3.8`, `pandas>=2.0`, and
`tqdm>=4.66`. The released post-hoc reducer only imports NumPy, pandas, and
Matplotlib and is CPU-capable. The repository provides lower bounds, not an
exact lock or wheel hashes; disposable build closure remains pending.

Root license evidence explicitly calls code Apache-2.0 and paper material
CC-BY-4.0. It does not explicitly name `model.pt` or frozen numeric outputs.
`artifact-inventory.json` therefore records checkpoint/frozen-artifact license
scope as unresolved. Author confirmation or a repository notice covering those
artifacts is required before the qualification `license_closure` gate can pass.

## Recreate ignored inputs

Run from the StressKit repository root. Existing targets should be verified,
not overwritten.

```bash
STRESSKIT_REPO=/Users/arth_rogerthat/Downloads/impact/Stresskit
SOURCE_ROOT="$STRESSKIT_REPO/.stresskit/upstreams"
SOURCE_PATH="$SOURCE_ROOT/mechanistic_tomography"
RAW_ROOT="$STRESSKIT_REPO/.stresskit/intake/mechtomo_finite_effect_map_recovery/raw"
EXTRACTED_ROOT="$STRESSKIT_REPO/.stresskit/intake/mechtomo_finite_effect_map_recovery/extracted"
INTAKE_DIR="$STRESSKIT_REPO/benchmark/intake/mechtomo_finite_effect_map_recovery"

mkdir -p "$SOURCE_ROOT" "$RAW_ROOT" "$EXTRACTED_ROOT"
git clone --filter=blob:none --no-checkout \
  https://github.com/kwisatzh/mechanistic-tomography "$SOURCE_PATH"
git -C "$SOURCE_PATH" fetch --depth=1 origin \
  5c097d2175c632a4a15e359cb4d94ec923168472
git -C "$SOURCE_PATH" checkout --detach \
  5c097d2175c632a4a15e359cb4d94ec923168472

test "$(git -C "$SOURCE_PATH" rev-parse HEAD)" = \
  5c097d2175c632a4a15e359cb4d94ec923168472
test "$(git -C "$SOURCE_PATH" rev-parse 'HEAD^{tree}')" = \
  261ed565077cb55dc61cc004fd1bd7ef9be01c67
test -z "$(git -C "$SOURCE_PATH" status --porcelain)"

git -C "$SOURCE_PATH" archive --format=tar \
  --prefix=mechanistic-tomography/ \
  -o "$STRESSKIT_REPO/.stresskit/intake/mechtomo_finite_effect_map_recovery/mechanistic-tomography-5c097d2.tar" \
  5c097d2175c632a4a15e359cb4d94ec923168472

curl --fail --location --retry 3 \
  --output "$RAW_ROOT/arxiv-2608.19338v1.pdf" \
  https://arxiv.org/pdf/2608.19338v1
curl --fail --location --retry 3 \
  --output "$RAW_ROOT/arxiv-2608.19338v1-source.tar.gz" \
  https://export.arxiv.org/e-print/2608.19338v1

tar -tzf "$RAW_ROOT/arxiv-2608.19338v1-source.tar.gz"
mkdir -p "$RAW_ROOT/arxiv-2608.19338v1-source"
tar -xzf "$RAW_ROOT/arxiv-2608.19338v1-source.tar.gz" \
  -C "$RAW_ROOT/arxiv-2608.19338v1-source" \
  --no-same-owner --no-same-permissions

pdftotext -layout -nopgbrk -enc UTF-8 \
  "$RAW_ROOT/arxiv-2608.19338v1.pdf" \
  "$EXTRACTED_ROOT/arxiv-2608.19338v1.txt"
stresskit audit extract-source \
  "$RAW_ROOT/arxiv-2608.19338v1-source/nt_mi_control_position_v21-v2.tex" \
  -o "$EXTRACTED_ROOT/nt_mi_control_position_v21-v2.tex.txt"
stresskit audit extract-source \
  "$SOURCE_PATH/experiments/hmm/sparse_tomography_posthoc.py" \
  -o "$EXTRACTED_ROOT/sparse_tomography_posthoc.py.txt"
```

PDF extraction is frozen to Poppler `pdftotext` 26.01.0 with `-layout
-nopgbrk -enc UTF-8`. `source-intake.json` rejects any changed raw or extracted
bytes.

## Rebuild SourceBundle and closure

```bash
stresskit audit source "$INTAKE_DIR/source-intake.json" \
  --cas "$STRESSKIT_REPO/.stresskit/cas" \
  --closure-input sha256:0e1cec96c5c4f0b1ba7d3a538af48f787e50992fd7b7c7eb00ca3af82f0d3aec="$INTAKE_DIR/artifact-inventory.json" \
  --closure-input sha256:d7558bbb4996e8cec312ebb8bb0fd9bc21e4883848f4a09e8575310cd10fd3d9="$INTAKE_DIR/claim-query.txt" \
  --closure-input sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30="$SOURCE_PATH/LICENSE" \
  --closure-input sha256:48e4dc03652e5c1bc771ded0d56b3bcd182b11440566a3a93c1566a0fdcd2391="$SOURCE_PATH/LICENSE-PAPER.md" \
  --closure-input sha256:d79efb21a7b4e60b3039e4d389cd0bc7d9cfe3f928af33da9230a747fb1f1e4d="$SOURCE_PATH/NOTICE" \
  --closure-input sha256:4d8689f8615cd2e78972f46dc022ae6b11d02eb2a4430bbae6cc013b0f299983="$SOURCE_PATH/experiments/hmm/frozen/model.pt" \
  --closure-input sha256:3836f7b1cc1815430c0fa042323ff2922ab2d536d764709e7e5e5ffdbac51b5d="$SOURCE_PATH/experiments/hmm/frozen/config.json" \
  --closure-input sha256:a59abe984e70a904185d8c5a3e295bcdf89d70f2da9a1fde1959b1061e0dc8bd="$SOURCE_PATH/experiments/hmm/frozen/nt_mi_set1_v2/measurement_matrix_A.npy" \
  --closure-input sha256:fe9b58ac7738b3a78367df35ab9ca816e5d44695f9ba57ec1dee88571b4393c2="$SOURCE_PATH/experiments/hmm/frozen/nt_mi_set1_v2/tomography_measurements.csv" \
  --closure-input sha256:15096074eb36d6a9fba855fa366abe0b28b7646d1899cb1d46a4ac23bb05d4bc="$SOURCE_PATH/experiments/hmm/frozen/nt_mi_set1_v2/direct_mi_z1.npy" \
  --closure-input sha256:212e1876e7f79e123d5867191de6bcd78b3cf364a1c2ad96a0d4c21972d75cdc="$SOURCE_PATH/experiments/hmm/frozen/nt_mi_sparse_v1/sparse_recovery_sample_efficiency.csv" \
  --closure-input sha256:f54e519eb6432fe2ee7c20ca13dec2dccdd7c7a61beede8a6d8109bdb38c3f25="$SOURCE_PATH/experiments/hmm/frozen/nt_mi_sparse_v1/sparse_recovery_summary.json" \
  --closure-input sha256:e53aa11056e971022201d4f48342514261a60a057995d75c6dd3e798707a7db0="$SOURCE_PATH/experiments/hmm/nt_mi_correspondence.py" \
  --closure-input sha256:3f69529a4833de87813b4b562ddc00c558e3ce2283de21bb77f544496ca03b3b="$SOURCE_PATH/experiments/hmm/hmm_observer_control.py" \
  --closure-output "$INTAKE_DIR/source-closure.json" \
  -o "$INTAKE_DIR/source-bundle.json"
```

Verified SourceBundle semantic digest:
`sha256:79cfe48ad882cb8b3cbe0035029086846b4ba0a0c307c40fd2f1b24546aa4174`.
The closure contains 33 objects and 46,863,731 bytes.

## Remaining gates

This candidate is now excluded from the current release after its frozen agent
panel abstained. Independent blockers remain: checkpoint/frozen-output license
scope is unresolved, dependency wheels are not locked, and no isolated CPU
execution smoke exists. Those blockers were not bypassed and no ResourcePlan
was issued.

## Live panel execution

After intake, StressKit froze three distinct OpenRouter routes: xAI Grok 4.6
and Mistral-hosted GLM 5.2 as extractors, then Google Gemini 3.1 Pro as the
dependency-gated critic. Extractor A returned byte-valid support anchors.
Extractor B returned four quotes absent byte-for-byte from its declared source.
StressKit rejected that response before constructing an `AgentOpinion`, did not
run the critic, did not retry, and published `abstain`.

`panel-execution.json`, `discovery-decision.json`, and
`panel-abstention.json` preserve exact slot accounting. The panel remains bound
to its original `source-bundle.panel-v1.json` and
`source-closure.panel-v1.json`; later outcome-blind code-map corrections created
the current SourceBundle and did not rewrite live model provenance. The
qualification ledger verifies the old bundle, accepted opinion, rejected raw
response, four absent quote bytes, routes, and 55-object CAS closure before
recording the row as excluded.

This is an agent-evidence validation result, not a reproduction or finding about
whether the paper's scientific claim is true. No GPU, ClaimRecord, AuditSpec,
ResourcePlan, upstream experiment, or StressKit experiment was run.
