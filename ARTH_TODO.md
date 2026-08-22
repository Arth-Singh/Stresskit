<!-- ⚠️ NOTE TO SELF (Arth): remove this file before the final code audit —
     don't wanna make this public on twitter lol.
     (It's already public the moment it's pushed, so nothing sensitive lives
     here — task names only, no tokens, no credentials, ever.) -->

# Arth's launch checklist

Things only Arth can do (credentials, repo settings, GPU box). Everything
here is safe-public — actual secrets go in a password manager, never in
this repo.

## Blocking the launch

- [ ] **PyPI release** — create/log into the PyPI account, add a scoped API
      token locally (`~/.pypirc` or keychain — NOT in this repo), then:
      `python -m pip install build twine && python -m build && twine upload dist/*`.
      Name is `stress-kit` (already set in pyproject). After it's live:
      update the README install line, the `stresskit` default in
      `action.yml`, and drop the "PyPI release pending" footnote.
- [ ] **Merge `claude/new-session-4337y6` → `main`** (open a PR so CI runs
      the new audit gate once before merge).
- [ ] **GPU runs** (when off the plane): Tier 1 of
      [`references/TARGETS.md`](references/TARGETS.md) — start with
      `python references/run_ioi_gpt2_card.py --model gpt2-xl`, then the
      refusal-direction card (highest-visibility next result).

## Repo settings (5 minutes in the GitHub UI)

- [ ] Enable **Discussions** (the issue-template config already links to it).
- [ ] Enable **GitHub Pages**: Settings → Pages → Source: "GitHub Actions".
      The `pages.yml` workflow then publishes the results site
      (index + card pages + trace charts) on every push to main —
      it will appear at arth-singh.github.io/Stresskit/. Once live, link it
      from the README's Reference batteries section and the repo About box.
- [ ] Add topics: `interpretability`, `mechanistic-interpretability`,
      `reproducibility`, `sparse-autoencoders`, `stability`.
- [ ] Set a social-preview image (a screenshot of an HTML card render is
      perfect: `stresskit render references/cards/ioi_gpt2_small.json --html`).
- [ ] Protect `main`: require the CI checks (test, audit-cards,
      test-minimal-deps) to pass.

## Before publicizing the C/D grades (PROTOCOL §6 — 14-day window)

- [ ] Courtesy issue on `adamkarvonen/activation_oracles` linking the three
      oracle cards + runner + PROTOCOL.
- [ ] Courtesy issue on `anthropics/jacobian-lens` for the J-lens card.
- [ ] Link (or note the absence of) their responses from the cards' notes.

## Nice to have

- [ ] Zenodo↔GitHub integration for a citable DOI (then add it to
      CITATION.cff and the README bibtex).
- [ ] Colab notebook reproducing the IOI card on a free GPU (the "run the
      notebook" answer to any dispute).
- [ ] Tag a release so the GitHub Action can be pinned as `@v0` and
      submitted to the Actions marketplace.
- [ ] Launch write-up + thread (Claude has offered to draft; the headline
      is the specificity result, per references/HYPOTHESES.md H2).
