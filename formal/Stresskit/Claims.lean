import Mathlib.Data.Fin.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace Stresskit

/-- Number of unordered distinct-run pairs with different categorical claims. -/
def pairDisagreementCount [DecidableEq κ] {n : Nat} (labels : Fin n → κ) : Nat :=
  ∑ i : Fin n, ∑ j : Fin n, if i < j ∧ labels i ≠ labels j then 1 else 0

/-- Exact categorical flip rate; unavailable with fewer than two runs. -/
def flipRate [DecidableEq κ] {n : Nat} (labels : Fin n → κ) : Option ℚ :=
  if n < 2 then
    none
  else
    some (pairDisagreementCount labels / n.choose 2)

@[simp]
theorem flipRate_empty [DecidableEq κ] (labels : Fin 0 → κ) :
    flipRate labels = none := by
  simp [flipRate]

@[simp]
theorem flipRate_singleton [DecidableEq κ] (labels : Fin 1 → κ) :
    flipRate labels = none := by
  simp [flipRate]

theorem pairDisagreementCount_constant [DecidableEq κ] {n : Nat} (label : κ) :
    pairDisagreementCount (fun _ : Fin n => label) = 0 := by
  simp [pairDisagreementCount]

/-- Largest class count from a sufficient-statistic vector. -/
def modalCount (counts : List Nat) : Nat := counts.foldr max 0

/-- Exact modal share; unavailable when no observations were counted. -/
def modalShare (counts : List Nat) : Option ℚ :=
  if counts.sum = 0 then none else some (modalCount counts / counts.sum)

/-- Filability is a declared tolerance rule, not a universal constant. -/
def filable (counts : List Nat) (alpha : ℚ) : Bool :=
  match modalShare counts with
  | none => false
  | some share => decide (share ≥ 1 - alpha)

theorem filable_iff (counts : List Nat) (alpha : ℚ) :
    filable counts alpha = true ↔
      ∃ share, modalShare counts = some share ∧ share ≥ 1 - alpha := by
  unfold filable
  cases h : modalShare counts with
  | none => simp
  | some share => simp

end Stresskit
