import Mathlib.Data.Finset.Card
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace Stresskit

/-- Exact rational Jaccard agreement, including StressKit's empty-pair rule. -/
def jaccard [DecidableEq α] (a b : Finset α) : ℚ :=
  if (a ∪ b).card = 0 then
    1
  else
    (a ∩ b).card / (a ∪ b).card

@[simp]
theorem jaccard_empty_empty [DecidableEq α] :
    jaccard (∅ : Finset α) ∅ = 1 := by
  simp [jaccard]

@[simp]
theorem jaccard_self [DecidableEq α] (a : Finset α) :
    jaccard a a = 1 := by
  simp [jaccard]

theorem jaccard_comm [DecidableEq α] (a b : Finset α) :
    jaccard a b = jaccard b a := by
  simp only [jaccard, Finset.union_comm a b, Finset.inter_comm a b]

theorem jaccard_nonnegative [DecidableEq α] (a b : Finset α) :
    0 ≤ jaccard a b := by
  simp only [jaccard]
  split_ifs
  · norm_num
  · positivity

theorem jaccard_le_one [DecidableEq α] (a b : Finset α) :
    jaccard a b ≤ 1 := by
  simp only [jaccard]
  split_ifs with h
  · norm_num
  · have hsub : a ∩ b ⊆ a ∪ b := by
      intro x hx
      exact Finset.mem_union.mpr (Or.inl (Finset.mem_inter.mp hx).1)
    have hcard : (a ∩ b).card ≤ (a ∪ b).card := Finset.card_le_card hsub
    have hden : (0 : ℚ) < (a ∪ b).card := by
      exact_mod_cast Nat.pos_of_ne_zero h
    exact (div_le_one hden).2 (by exact_mod_cast hcard)

end Stresskit
