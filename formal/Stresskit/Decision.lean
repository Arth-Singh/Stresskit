import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace Stresskit

inductive Direction where
  | atLeast
  | atMost
  deriving DecidableEq, Repr

inductive DecisionState where
  | pass
  | fail
  | inconclusive
  deriving DecidableEq, Repr

/-- Three-state interval decision used by confirmatory StressKit. -/
def decideInterval
    (direction : Direction)
    (lo hi threshold : ℚ)
    (intervalAvailable minimumNMet : Bool) : DecisionState :=
  if intervalAvailable = false ∨ minimumNMet = false then
    .inconclusive
  else
    match direction with
    | .atLeast =>
        if threshold ≤ lo then .pass
        else if hi < threshold then .fail
        else .inconclusive
    | .atMost =>
        if hi ≤ threshold then .pass
        else if threshold < lo then .fail
        else .inconclusive

@[simp]
theorem atLeast_pass_iff (lo hi threshold : ℚ) :
    decideInterval .atLeast lo hi threshold true true = .pass ↔ threshold ≤ lo := by
  by_cases h₁ : threshold ≤ lo
  · simp [decideInterval, h₁]
  · by_cases h₂ : hi < threshold
    · simp [decideInterval, h₁, h₂]
    · simp [decideInterval, h₁, h₂]

@[simp]
theorem atMost_pass_iff (lo hi threshold : ℚ) :
    decideInterval .atMost lo hi threshold true true = .pass ↔ hi ≤ threshold := by
  by_cases h₁ : hi ≤ threshold
  · simp [decideInterval, h₁]
  · by_cases h₂ : threshold < lo
    · simp [decideInterval, h₁, h₂]
    · simp [decideInterval, h₁, h₂]

@[simp]
theorem unavailable_is_inconclusive
    (direction : Direction) (lo hi threshold : ℚ) (minimumNMet : Bool) :
    decideInterval direction lo hi threshold false minimumNMet = .inconclusive := by
  simp [decideInterval]

@[simp]
theorem underpowered_is_inconclusive
    (direction : Direction) (lo hi threshold : ℚ) (intervalAvailable : Bool) :
    decideInterval direction lo hi threshold intervalAvailable false = .inconclusive := by
  simp [decideInterval]

/-- Required-check aggregation: one failure is a validity-gate failure. -/
def combineChecks (states : List DecisionState) : DecisionState :=
  if states.contains .fail then
    .fail
  else if states ≠ [] ∧ states.all (· == .pass) then
    .pass
  else
    .inconclusive

theorem combineChecks_fails_if_any (states : List DecisionState)
    (h : .fail ∈ states) : combineChecks states = .fail := by
  simp [combineChecks, h]

@[simp]
theorem combineChecks_all_pass (n : Nat) :
    combineChecks (List.replicate (n + 1) .pass) = .pass := by
  simp [combineChecks]

@[simp]
theorem combineChecks_empty : combineChecks [] = .inconclusive := by
  simp [combineChecks]

end Stresskit
