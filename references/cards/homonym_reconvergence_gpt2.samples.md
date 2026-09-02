# Randomly selected raw examples (base configuration)

Selected with `random.Random(0)`, not cherry-picked. Activation distance = cosine distance between the two residuals at the read position; KL = symmetrised KL between the raw-lens next-token distributions. Layers 0..L-1 are post-block residuals.

## Homonyms (454 items)

Aggregate curves (mean over words of per-word means):

| layer | activation | logit | KL |
|---|---|---|---|
| 0 | 0.0182 | 0.0368 | 0.260 |
| 1 | 0.0335 | 0.0529 | 0.649 |
| 2 | 0.0823 | 0.1277 | 2.578 |
| 3 | 0.1401 | 0.2028 | 5.088 |
| 4 | 0.1760 | 0.2521 | 7.191 |
| 5 | 0.2018 | 0.2849 | 9.067 |
| 6 | 0.2205 | 0.3171 | 11.039 |
| 7 | 0.2325 | 0.3501 | 12.774 |
| 8 | 0.2429 | 0.3812 | 14.517 |
| 9 | 0.2381 | 0.4161 | 15.991 |
| 10 | 0.1647 | 0.4380 | 16.857 |
| 11 | 0.0955 | 0.4183 | 17.315 |

**trunk** (token `' trunk'`, read offset 0)
- A: The car trunk holds luggage. (position 3)
- B: The tree trunk grows tall. (position 3)
- activation distance by layer: [0.039, 0.05, 0.074, 0.083, 0.084, 0.065, 0.056, 0.045, 0.041, 0.037, 0.022, 0.014]
- KL by layer: [0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 1.13, 0.08, 6.91, 5.42, 0.17, 12.21]
- item peak layer 4, KL argmax 11

**organ** (token `' organ'`, read offset 0)
- A: The heart is a vital organ. (position 6)
- B: She plays the organ at church on Sundays. (position 4)
- activation distance by layer: [0.009, 0.014, 0.051, 0.105, 0.145, 0.18, 0.177, 0.174, 0.193, 0.202, 0.142, 0.074]
- KL by layer: [0.0, 0.0, 0.0, 0.29, 0.56, 1.59, 10.42, 12.18, 14.69, 16.69, 18.42, 18.42]
- item peak layer 9, KL argmax 10

**pitch** (token `' pitch'`, read offset 0)
- A: The baseball player has a great pitch. (position 7)
- B: The ship rolled in the pitch and roll of the waves. (position 6)
- activation distance by layer: [0.035, 0.058, 0.107, 0.147, 0.176, 0.195, 0.202, 0.198, 0.184, 0.176, 0.138, 0.106]
- KL by layer: [0.0, 0.0, 4.72, 0.02, 0.02, 0.07, 2.19, 0.11, 0.0, 3.09, 1.17, 15.42]
- item peak layer 6, KL argmax 11

**sow** (token `' sow'`, read offset 0)
- A: The farmer will sow seeds in spring. (position 4)
- B: We need to sow discord among our enemies. (position 4)
- activation distance by layer: [0.013, 0.023, 0.035, 0.044, 0.061, 0.069, 0.088, 0.091, 0.117, 0.116, 0.091, 0.049]
- KL by layer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 1.79, 0.0, 0.0, 0.0, 1.31]
- item peak layer 8, KL argmax 7

**bass** (token `' bass'`, read offset 0)
- A: He caught a large bass in the lake. (position 5)
- B: She sings bass in the choir. (position 3)
- activation distance by layer: [0.008, 0.014, 0.041, 0.065, 0.092, 0.102, 0.104, 0.114, 0.128, 0.135, 0.104, 0.073]
- KL by layer: [0.0, 0.0, 0.99, 2.9, 0.77, 0.1, 0.05, 8.01, 3.11, 12.84, 17.94, 18.42]
- item peak layer 9, KL argmax 11

## Polysemes (template) (115 items)

Aggregate curves (mean over words of per-word means):

| layer | activation | logit | KL |
|---|---|---|---|
| 0 | 0.0192 | 0.0383 | 0.183 |
| 1 | 0.0345 | 0.0556 | 0.404 |
| 2 | 0.0778 | 0.1251 | 1.558 |
| 3 | 0.1251 | 0.1861 | 3.465 |
| 4 | 0.1545 | 0.2283 | 4.951 |
| 5 | 0.1769 | 0.2587 | 6.591 |
| 6 | 0.1934 | 0.2921 | 8.638 |
| 7 | 0.2024 | 0.3182 | 10.583 |
| 8 | 0.2140 | 0.3535 | 13.214 |
| 9 | 0.2162 | 0.3934 | 14.686 |
| 10 | 0.1579 | 0.4249 | 16.226 |
| 11 | 0.0860 | 0.4131 | 17.002 |

**host** (token `' host'`, read offset 0)
- A: The gracious host welcomed guests into her home. (position 3)
- B: The web server host crashed during peak traffic hours. (position 4)
- activation distance by layer: [0.028, 0.039, 0.087, 0.132, 0.158, 0.174, 0.175, 0.161, 0.164, 0.179, 0.118, 0.07]
- KL by layer: [0.0, 0.0, 0.01, 0.01, 0.0, 0.3, 8.08, 18.22, 14.35, 13.41, 14.58, 18.42]
- item peak layer 9, KL argmax 11

**code** (token `' code'`, read offset 0)
- A: The spy used a secret code for messages. (position 6)
- B: She writes code in Python every day. (position 3)
- activation distance by layer: [0.03, 0.051, 0.104, 0.14, 0.163, 0.165, 0.169, 0.161, 0.165, 0.154, 0.117, 0.07]
- KL by layer: [0.08, 0.03, 0.32, 2.27, 3.51, 4.13, 2.35, 17.64, 17.86, 17.12, 18.42, 18.42]
- item peak layer 6, KL argmax 10

**icon** (token `' icon'`, read offset 0)
- A: The religious icon hung above the altar. (position 3)
- B: Click the icon to open the program. (position 3)
- activation distance by layer: [0.014, 0.029, 0.096, 0.197, 0.246, 0.275, 0.297, 0.317, 0.328, 0.337, 0.241, 0.114]
- KL by layer: [0.01, 0.0, 0.06, 0.44, 0.29, 0.66, 1.91, 16.08, 18.42, 18.42, 18.42, 18.42]
- item peak layer 9, KL argmax 8

## Permuted pairs (graded null) (454 items)

Aggregate curves (mean over words of per-word means):

| layer | activation | logit | KL |
|---|---|---|---|
| 0 | 0.3212 | 0.6525 | 18.356 |
| 1 | 0.4169 | 0.6471 | 18.358 |
| 2 | 0.4438 | 0.6783 | 18.351 |
| 3 | 0.4754 | 0.6899 | 18.317 |
| 4 | 0.4780 | 0.6801 | 18.277 |
| 5 | 0.4833 | 0.6756 | 18.362 |
| 6 | 0.4762 | 0.6645 | 18.344 |
| 7 | 0.4577 | 0.6648 | 18.369 |
| 8 | 0.4338 | 0.6560 | 18.359 |
| 9 | 0.3867 | 0.6512 | 18.307 |
| 10 | 0.2436 | 0.6298 | 18.297 |
| 11 | 0.1316 | 0.5647 | 18.297 |

**patient** (token `' patient| slip'`, read offset 0)
- A: The doctor examined the patient carefully. (position 5)
- B: She wrote a note on a slip of paper. (position 7)
- activation distance by layer: [0.323, 0.425, 0.446, 0.458, 0.467, 0.479, 0.464, 0.458, 0.422, 0.402, 0.303, 0.177]
- KL by layer: [18.41, 18.41, 18.4, 18.4, 18.4, 18.41, 18.41, 18.41, 18.42, 18.42, 18.42, 18.42]
- item peak layer 5, KL argmax 11

**refuse** (token `' refuse| clip'`, read offset 0)
- A: I refuse to accept these terms. (position 2)
- B: Could you clip the article from the newspaper. (position 3)
- activation distance by layer: [0.306, 0.413, 0.417, 0.498, 0.498, 0.496, 0.482, 0.45, 0.415, 0.352, 0.188, 0.128]
- KL by layer: [18.42, 18.42, 18.42, 18.41, 18.39, 18.41, 18.41, 18.41, 18.41, 18.42, 18.42, 18.42]
- item peak layer 3, KL argmax 11

**contract** (token `' contract| spell'`, read offset 0)
- A: Sign the contract before you start work. (position 3)
- B: We had a spell of cold weather last week. (position 4)
- activation distance by layer: [0.282, 0.428, 0.456, 0.481, 0.488, 0.472, 0.44, 0.439, 0.446, 0.383, 0.25, 0.153]
- KL by layer: [18.41, 18.41, 18.41, 18.42, 18.42, 18.41, 18.41, 18.41, 18.42, 18.41, 18.42, 18.42]
- item peak layer 4, KL argmax 11

## Sequence-order control (alternative null) (281 items)

Aggregate curves (mean over words of per-word means):

| layer | activation | logit | KL |
|---|---|---|---|
| 0 | 0.0061 | 0.0110 | 0.117 |
| 1 | 0.0099 | 0.0144 | 0.241 |
| 2 | 0.0197 | 0.0289 | 0.543 |
| 3 | 0.0325 | 0.0444 | 1.050 |
| 4 | 0.0454 | 0.0594 | 1.638 |
| 5 | 0.0550 | 0.0703 | 2.327 |
| 6 | 0.0593 | 0.0755 | 3.465 |
| 7 | 0.0611 | 0.0798 | 4.809 |
| 8 | 0.0685 | 0.0932 | 6.646 |
| 9 | 0.0729 | 0.1073 | 8.460 |
| 10 | 0.0548 | 0.1184 | 10.632 |
| 11 | 0.0324 | 0.1129 | 12.269 |

** book** (token `' book'`, read offset 0)
- A: She gave a book to him. (position 4)
- B: She gave him a book. (position 5)
- activation distance by layer: [0.001, 0.002, 0.004, 0.009, 0.013, 0.022, 0.034, 0.034, 0.041, 0.048, 0.039, 0.028]
- KL by layer: [0.04, 0.12, 0.29, 0.1, 0.25, 2.19, 1.24, 3.62, 0.22, 8.66, 6.41, 17.5]
- item peak layer 9, KL argmax 11

** photo** (token `' photo'`, read offset 0)
- A: I showed my friend the photo. (position 6)
- B: My friend showed me the photo. (position 6)
- activation distance by layer: [0.0, 0.001, 0.002, 0.005, 0.007, 0.008, 0.006, 0.006, 0.005, 0.006, 0.005, 0.004]
- KL by layer: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- item peak layer 5, KL argmax 0

** the** (token `' the'`, read offset 0)
- A: The hammer broke the window. (position 4)
- B: The window broke the hammer. (position 4)
- activation distance by layer: [0.001, 0.001, 0.004, 0.005, 0.014, 0.023, 0.032, 0.039, 0.046, 0.045, 0.024, 0.015]
- KL by layer: [0.02, 0.01, 0.01, 0.0, 0.07, 2.97, 1.38, 2.18, 12.62, 7.67, 12.12, 16.24]
- item peak layer 8, KL argmax 11

