import Stresskit.Core
import Stresskit.Decision
import Stresskit.RandomNull

namespace Stresskit

structure JaccardVector where
  left : Finset Nat
  right : Finset Nat
  expected : ℚ

def jaccardVectors : List JaccardVector := [
  ⟨∅, ∅, 1⟩,
  ⟨{1}, {2}, 0⟩,
  ⟨{1, 2}, {2, 3}, 1 / 3⟩,
  ⟨{1, 2, 3}, {1, 2, 3}, 1⟩
]

theorem jaccardVectors_conform :
    jaccardVectors.all (fun v => jaccard v.left v.right == v.expected) = true := by
  native_decide

structure RandomNullVector where
  universeSize : Nat
  leftSize : Nat
  rightSize : Nat
  expected : ℚ

def randomNullVectors : List RandomNullVector := [
  ⟨3, 1, 1, 1 / 3⟩,
  ⟨3, 1, 2, 1 / 3⟩,
  ⟨5, 2, 2, 3 / 10⟩,
  ⟨10, 2, 3, 29 / 180⟩,
  ⟨20, 4, 4, 999 / 8075⟩,
  ⟨144, 15, 15,
    2584850149088656364382653 / 45638295405532475996009088⟩
]

theorem randomNullVectors_conform :
    randomNullVectors.all (fun v =>
      exactExpectedRandomJaccard v.universeSize v.leftSize v.rightSize ==
        v.expected) = true := by
  native_decide

def decisionVectors : List Bool := [
  decideInterval .atLeast (81 / 100) (95 / 100) (8 / 10) true true == .pass,
  decideInterval .atLeast (6 / 10) (79 / 100) (8 / 10) true true == .fail,
  decideInterval .atLeast (7 / 10) (9 / 10) (8 / 10) true true == .inconclusive,
  decideInterval .atMost (5 / 100) (2 / 10) (2 / 10) true true == .pass,
  decideInterval .atMost (5 / 100) (1 / 10) (2 / 10) true false == .inconclusive
]

theorem decisionVectors_conform : decisionVectors.all id = true := by
  native_decide

end Stresskit
