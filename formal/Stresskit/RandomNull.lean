import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace Stresskit

/-- Hypergeometric mass as an exact rational. Inputs are finite-cardinality data. -/
def hypergeomProbability (universeSize leftSize rightSize intersection : Nat) : ℚ :=
  (leftSize.choose intersection *
      (universeSize - leftSize).choose (rightSize - intersection)) /
    universeSize.choose rightSize

/-- Exact expected Jaccard for independent uniform fixed-size subsets. -/
def exactExpectedRandomJaccard
    (universeSize leftSize rightSize : Nat) : ℚ :=
  if leftSize = 0 then
    if rightSize = 0 then 1 else 0
  else if rightSize = 0 then
    0
  else
    ∑ intersection ∈ Finset.Icc
        (leftSize + rightSize - universeSize)
        (min leftSize rightSize),
      (intersection / (leftSize + rightSize - intersection) : ℚ) *
        hypergeomProbability universeSize leftSize rightSize intersection

@[simp]
theorem exactExpectedRandomJaccard_empty_empty (universeSize : Nat) :
    exactExpectedRandomJaccard universeSize 0 0 = 1 := by
  simp [exactExpectedRandomJaccard]

@[simp]
theorem exactExpectedRandomJaccard_empty_left
    (universeSize rightSize : Nat) (h : rightSize ≠ 0) :
    exactExpectedRandomJaccard universeSize 0 rightSize = 0 := by
  simp [exactExpectedRandomJaccard, h]

theorem exactExpectedRandomJaccard_3_1_1 :
    exactExpectedRandomJaccard 3 1 1 = 1 / 3 := by
  native_decide

theorem exactExpectedRandomJaccard_3_1_2 :
    exactExpectedRandomJaccard 3 1 2 = 1 / 3 := by
  native_decide

theorem ratioOfExpectations_is_not_exact :
    exactExpectedRandomJaccard 5 2 2 ≠ (2 : ℚ) / (2 * 5 - 2) := by
  native_decide

end Stresskit
