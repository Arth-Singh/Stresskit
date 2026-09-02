# Randomly selected raw examples (base run)

Selected with `random.Random(0)` from the 60 statements the probes see, not cherry-picked. Model responses are greedy, first 48 new tokens, upstream template. Decision values are the held-out probe scores at depth 16 (positive = the probe fires; the truth probe is only applied to true/false statements, its own contrast).

## mod_penguin_improbable (improbable; expected label `coherent`)

**Statement.** A penguin wandered one hundred kilometers inland and returned to the colony unharmed.
- model label: `coherent` -- response: `'**Coherent**\n\nThe statement is coherent because it describes a plausible, albeit unusual, sequence of events – a penguin traveling a significant distance and returning safely, which aligns with the known capabilities and behaviors of penguins.'`
- impossibility probe decision: -4.88
- truth probe: not applied (outside its contrast)

## mod_apple_true (true; expected label `coherent`)

**Statement.** Apples grow on trees.
- model label: `coherent` -- response: `'**Coherent**\n\nThis statement is coherent because it presents a generally accepted and understandable relationship between apples and trees, forming a logical connection.'`
- impossibility probe decision: -7.37
- truth probe decision: -5.58

## mod_gold_true (true; expected label `coherent`)

**Statement.** Gold is a metal that does not rust.
- model label: `contradiction` -- response: `'**Contradiction**\n\nThe statement contradicts established scientific knowledge, as gold *does* react with certain chemicals under specific conditions and can form tarnish, effectively behaving like it "rusts."'`
- impossibility probe decision: -1.90
- truth probe decision: -0.07

## mod_train_true (true; expected label `coherent`)

**Statement.** Trains run on rails.
- model label: `coherent` -- response: `'**Coherent**\n\nThis statement is coherent because it expresses a generally accepted and understandable relationship between trains and rails – trains are designed to operate along fixed tracks.'`
- impossibility probe decision: -4.14
- truth probe decision: -7.05

## mod_apple_improbable (improbable; expected label `coherent`)

**Statement.** Every apple in the orchard fell to the ground during the same minute.
- model label: `contradiction` -- response: `'**Contradiction**\n\nThe statement is a contradiction because it’s impossible for every single apple in an orchard to fall to the ground simultaneously due to the physical limitations of gravity and the size of the orchard. \n\nWould you like'`
- impossibility probe decision: -2.85
- truth probe: not applied (outside its contrast)

## mod_paris_improbable (improbable; expected label `coherent`)

**Statement.** Every train into Paris arrived exactly on time for an entire month.
- model label: `contradiction` -- response: `'**Contradiction**\n\nThis statement is a contradiction because it asserts an impossible event – trains arriving precisely on time every single day for a month, given the inherent complexities and potential delays in train travel.'`
- impossibility probe decision: -3.64
- truth probe: not applied (outside its contrast)

## mod_whale_true (true; expected label `coherent`)

**Statement.** Whales are mammals that breathe air.
- model label: `coherent` -- response: `'**Coherent**\n\nThe statement is coherent because it presents a logically consistent and widely accepted fact about whales – they are mammals and mammals breathe air. \n\nWould you like me to classify another statement?'`
- impossibility probe decision: -6.87
- truth probe decision: -3.61

## mod_triangle_true (true; expected label `coherent`)

**Statement.** A triangle has exactly three sides.
- model label: `coherent` -- response: `'**Coherent**\n\nThe statement is coherent because it presents a simple, logically sound definition of a triangle – a closed shape formed by three straight line segments – and is universally accepted as true.'`
- impossibility probe decision: -3.14
- truth probe decision: -6.21

