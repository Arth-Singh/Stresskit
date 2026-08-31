# Stability Scoreboard

Every finding graded by StressKit's reference batteries, under the
default thresholds and the protocol in
[`references/PROTOCOL.md`](references/PROTOCOL.md). Each row links
to the full card; every card re-derives from its own recorded
metrics via `stresskit verify` (CI enforces this on every push).

A grade is a reliability measurement under pre-registered checks —
**not** a judgment of a paper's value, and never a claim of
misconduct. Undecided checks (95% CI straddling its bar) lower a
verdict's confidence; a low-confidence grade is provisional.

| finding | method | grade | confidence | checks passed | runs/answers | headline | card |
|---|---|---|---|---|---|---|---|
| IOI (ABC corruption) / gpt2 | head-level attribution patching, top-k by \|attribution\| | 🟢 **A** | **low** | 5/5 (2 undecided) | 45 | J=0.83, specificity 1.54× | [stability card](references/cards/ioi_gpt2_small.md) |
| IOI (ABC corruption) / gpt2-large | head-level attribution patching, top-k by \|attribution\| | 🟢 **A** | **low** | 5/5 (2 undecided) | 45 | J=0.85, specificity 1.63× | [stability card](references/scale/ioi_gpt2_large.md) |
| IOI (ABC corruption) / gpt2-medium | head-level attribution patching, top-k by \|attribution\| | 🟢 **A** | high | 5/5 | 45 | J=0.95, specificity 2.32× | [stability card](references/scale/ioi_gpt2_medium.md) |
| Greater-Than (YY-\>01 corruption) / gpt2-small | head-level attribution patching, top-k by \|attribution\| | 🟡 **B** | high | 4/5 | 45 | J=0.89, specificity 1.15× | [stability card](references/cards/greater_than_gpt2_small.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-0.8B | jlens transport, upstream hit criterion | 🟡 **B** | **low** | 3/5 (1 undecided) | 48 | J=0.46, specificity 0.86× | [stability card](references/h200-results/lens_baseline_jlens_qwen3p5_0p8b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-27B | logit transport, upstream hit criterion | 🟡 **B** | **low** | 3/5 (2 undecided) | 48 | J=0.48, specificity 0.82× | [stability card](references/h200-results/lens_baseline_logit_qwen3p5_27b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-4B | jlens transport, upstream hit criterion | 🟡 **B** | **low** | 3/5 (1 undecided) | 48 | J=0.47, specificity 0.89× | [stability card](references/h200-results/lens_baseline_jlens_qwen3p5_4b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.6-27B | logit transport, upstream hit criterion | 🟡 **B** | **low** | 3/5 (1 undecided) | 48 | J=0.48, specificity 0.64× | [stability card](references/h200-results/lens_baseline_logit_qwen3p6_27b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-0.8B | logit transport, upstream hit criterion | 🟠 **C** | **low** | 2/5 (1 undecided) | 48 | J=0.49, specificity 0.94× | [stability card](references/h200-results/lens_baseline_logit_qwen3p5_0p8b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-27B | jlens transport, upstream hit criterion | 🟠 **C** | **low** | 2/5 (1 undecided) | 48 | J=0.47, specificity 0.93× | [stability card](references/h200-results/lens_baseline_jlens_qwen3p5_27b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-4B | Jacobian lens (pre-fitted, n=1000), upstream hit criterion | 🟠 **C** | **low** | 2/5 (2 undecided) | 20 | J=0.45, specificity 0.78× | [stability card](references/cards/jlens_qwen3p5_4b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-4B | logit transport, upstream hit criterion | 🟠 **C** | **low** | 2/5 (1 undecided) | 48 | J=0.48, specificity 0.81× | [stability card](references/h200-results/lens_baseline_logit_qwen3p5_4b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.5-4B | tuned transport, upstream hit criterion | 🟠 **C** | **low** | 2/5 (1 undecided) | 48 | J=0.46, specificity 0.86× | [stability card](references/h200-results/lens_baseline_tuned_qwen3p5_4b.md) |
| lens-eval-multihop (vs association) / Qwen/Qwen3.6-27B | jlens transport, upstream hit criterion | 🟠 **C** | **low** | 2/5 (1 undecided) | 48 | J=0.46, specificity 0.87× | [stability card](references/h200-results/lens_baseline_jlens_qwen3p6_27b.md) |
| adamkarvonen/checkpoints\_latentqa\_only\_addition\_Qwen3-8B (Qwen3-8B) | activation reader | 🟠 **C** | high | 1/4 | 225 | accuracy 0.09, null hallucination 0.89 | [oracle report](references/cards/ao_qwen3_latentqa-only.md) |
| adamkarvonen/checkpoints\_cls\_only\_addition\_Qwen3-8B (Qwen3-8B) | activation reader | 🔴 **D** | high | 0/4 | 225 | accuracy 0.24, null hallucination 0.93 | [oracle report](references/cards/ao_qwen3_cls-only.md) |
| adamkarvonen/checkpoints\_latentqa\_cls\_past\_lens\_addition\_Qwen3-8B (Qwen3-8B) | activation reader | 🔴 **D** | high | 0/4 | 225 | accuracy 0.45, null hallucination 0.94 | [oracle report](references/cards/ao_qwen3_full-mixture.md) |

**Grades**: A — all applicable checks pass · B — at least half ·
C — at least one · D — none, or indistinguishable from random.

Want a finding on this board? See
[CONTRIBUTING.md](CONTRIBUTING.md) — submissions arrive as PRs
carrying the card JSON, the runner script, and a `stresskit verify`
pass.

*Generated by `stresskit scoreboard` — do not edit by hand.*
