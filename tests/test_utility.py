"""Downstream-utility axis: the non-internals baseline is mandatory, the
margin is graded against its interval, and an unanswered axis stays visible.
"""

import json

import pytest

from stresskit.card import StabilityCard, validate_card_dict
from stresskit.utility import (
    Baseline,
    attach_utility,
    best_non_internals,
    bootstrap_delta_ci,
    interpretability_phrasing,
    utility_block,
    utility_check,
    validate_utility_block,
)


PROMPTING = Baseline("keyword rules over the reply text", 0.44, uses_internals=False)
LOGIT_LENS = Baseline("logit lens on the final token", 0.66, uses_internals=True)
TASK = "flag support replies that contradict the order record"


def make_block(**over):
    kwargs = dict(
        task=TASK,
        metric="precision at 50 flags",
        with_method=0.71,
        baselines=[PROMPTING, LOGIT_LENS],
        n=4,
        paired_deltas=[0.30, 0.24, 0.28, 0.26],
    )
    kwargs.update(over)
    return utility_block(**kwargs)


class TestNonInternalsBaseline:
    def test_a_utility_claim_without_one_is_rejected(self):
        with pytest.raises(ValueError, match="uses_internals=False"):
            utility_block(task=TASK, metric="precision", with_method=0.71,
                          baselines=[LOGIT_LENS], n=4)

    def test_the_strongest_non_internals_baseline_is_the_reference(self):
        weak = Baseline("always flag", 0.10, uses_internals=False)
        assert best_non_internals([weak, PROMPTING, LOGIT_LENS]) is PROMPTING

    def test_an_internals_baseline_never_becomes_the_reference(self):
        block = make_block()
        assert block["reference_baseline"] == PROMPTING.name
        assert block["delta_vs_non_internals"] == pytest.approx(0.71 - 0.44)


class TestGrading:
    def test_margin_clear_of_the_bar_passes(self):
        assert utility_check(make_block())["state"] == "pass"

    def test_margin_at_or_below_the_bar_fails(self):
        block = make_block(with_method=0.40,
                           paired_deltas=[-0.04, -0.05, -0.03, -0.04])
        assert utility_check(block)["state"] == "fail"

    def test_a_straddling_interval_is_inconclusive_not_a_pass(self):
        block = make_block(with_method=0.45,
                           paired_deltas=[0.30, -0.28, 0.26, -0.24])
        assert utility_check(block)["state"] == "inconclusive"

    def test_without_an_interval_the_axis_cannot_be_resolved(self):
        block = make_block(paired_deltas=None)
        result = utility_check(block)
        assert result["state"] == "inconclusive"
        assert "paired_deltas" in result["reason"]

    def test_the_bar_can_be_raised_above_zero(self):
        block = make_block()
        assert utility_check(block, min_delta=0.5)["state"] == "fail"


class TestBootstrap:
    def test_interval_is_ordered_and_brackets_the_mean(self):
        deltas = [0.1, 0.2, 0.3, 0.4, 0.5]
        lo, hi = bootstrap_delta_ci(deltas, n_boot=500, seed=0)
        assert lo <= sum(deltas) / len(deltas) <= hi

    def test_same_seed_gives_the_same_interval(self):
        deltas = [0.1, 0.2, 0.3, 0.4]
        assert (bootstrap_delta_ci(deltas, n_boot=200, seed=7)
                == bootstrap_delta_ci(deltas, n_boot=200, seed=7))

    def test_one_item_cannot_support_an_interval(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_delta_ci([0.3])

    def test_paired_deltas_must_match_the_item_count(self):
        with pytest.raises(ValueError, match="but n is"):
            make_block(n=9)


class TestTaskPhrasing:
    def test_a_task_named_after_the_technique_is_flagged(self):
        block = make_block(task="raise SAE feature reconstruction fidelity")
        assert "task_phrasing_warning" in block
        assert "sparse autoencoder" in block["task_phrasing_warning"] or \
               "sae" in block["task_phrasing_warning"]

    def test_a_task_in_ordinary_language_is_not_flagged(self):
        assert "task_phrasing_warning" not in make_block()

    def test_phrasing_check_is_case_insensitive(self):
        assert "circuit" in interpretability_phrasing("Recover the CIRCUIT")

    def test_empty_task_is_rejected(self):
        with pytest.raises(ValueError, match="task statement"):
            make_block(task="   ")


class TestCardIntegration:
    def _card(self, utility=None):
        return StabilityCard(
            claim={"statement": "s"}, battery={"axes": ["seed"]},
            metrics={"pooled": {}},
            verdict={"grade": "A", "grade_rule": "v0.4",
                     "profile": "diagnostic",
                     "confirmatory_state": "not_applicable",
                     "required_checks": [], "checks": {},
                     "thresholds": {"random_floor": 1.5}},
            provenance={}, utility=utility,
        )

    def test_utility_survives_a_json_roundtrip(self):
        card = self._card(make_block())
        back = StabilityCard.from_dict(json.loads(json.dumps(card.to_dict())))
        assert back.utility == card.utility

    def test_a_card_without_utility_omits_the_key_entirely(self):
        assert "utility" not in self._card().to_dict()

    def test_an_unanswered_axis_renders_as_not_reported(self):
        md = self._card().to_markdown()
        assert "## Downstream utility" in md
        assert "NOT REPORTED" in md

    def test_an_answered_axis_renders_the_task_and_the_margin(self):
        md = self._card(make_block()).to_markdown()
        assert TASK in md
        assert PROMPTING.name in md
        assert "✅ pass" in md

    def test_a_card_carrying_only_internals_baselines_is_invalid(self):
        bad = make_block()
        bad["baselines"] = [LOGIT_LENS.to_dict()]
        with pytest.raises(ValueError, match="uses_internals=false"):
            validate_card_dict(self._card(bad).to_dict())

    def test_attach_validates_before_it_attaches(self):
        card = self._card()
        bad = make_block()
        bad["baselines"] = [LOGIT_LENS.to_dict()]
        with pytest.raises(ValueError, match="uses_internals=false"):
            attach_utility(card, bad)
        assert card.utility is None

    def test_attach_puts_the_block_on_a_card_stress_already_built(self):
        card = attach_utility(self._card(), make_block())
        assert card.utility["reference_baseline"] == PROMPTING.name
        assert "✅ pass" in card.to_markdown()

    def test_malformed_interval_is_rejected(self):
        bad = make_block()
        bad["delta_ci95"] = [0.5, 0.1]
        with pytest.raises(ValueError, match="ordered"):
            validate_utility_block(bad)
