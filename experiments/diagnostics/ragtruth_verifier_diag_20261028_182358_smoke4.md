# RAGTruth verifier diagnostic — ragtruth_verifier_diag_20261028_182358_smoke4

Claims: 17 gold-hallucinated + -13 gold-unflagged (seeded sample). Diagnostic prevalence is not natural prevalence: compare TPR/FPR, read precision/F1 as conditional on this mix.

## Flag = broad

| Condition | TP | FN | FP | TN | TPR [Wilson 95%] | FPR [Wilson 95%] | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| A_original | 0 | 4 | 0 | 0 | 0.000 [0.000, 0.490] | n/a [n/a, n/a] | n/a | n/a |
| B_readable | 1 | 3 | 0 | 0 | 0.250 [0.046, 0.699] | n/a [n/a, n/a] | 1.000 | 0.400 |

By task (broad):

| Condition | Task | TP/FN | FP/TN | TPR | FPR |
|---|---|---|---|---|---|
| A_original | Data2txt | 0/4 | 0/0 | 0.000 | n/a |
| B_readable | Data2txt | 1/3 | 0/0 | 0.250 | n/a |

## Flag = strict

| Condition | TP | FN | FP | TN | TPR [Wilson 95%] | FPR [Wilson 95%] | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| A_original | 0 | 4 | 0 | 0 | 0.000 [0.000, 0.490] | n/a [n/a, n/a] | n/a | n/a |
| B_readable | 1 | 3 | 0 | 0 | 0.250 [0.046, 0.699] | n/a [n/a, n/a] | 1.000 | 0.400 |

By task (strict):

| Condition | Task | TP/FN | FP/TN | TPR | FPR |
|---|---|---|---|---|---|
| A_original | Data2txt | 0/4 | 0/0 | 0.000 | n/a |
| B_readable | Data2txt | 1/3 | 0/0 | 0.250 | n/a |

## Label counts

- A_original: {"HALLUCINATED": {"SUPPORTED": 4}, "UNFLAGGED": {}}
- B_readable: {"HALLUCINATED": {"SUPPORTED": 3, "CONFLICTED": 1}, "UNFLAGGED": {}}

## Latency (recorded per-call, seconds) and evidence size

- A_original: mean recorded latency 31.799s over 4 calls; mean evidence chars 1938
- B_readable: mean recorded latency 11.781s over 4 calls; mean evidence chars 1650

## Condition B split by whether the whole source was visible

- full_source_visible=True: n=4 TP=1 FN=3 FP=0 TN=0 TPR=0.250 FPR=n/a
- full_source_visible=False: n=0 TP=0 FN=0 FP=0 TN=0 TPR=n/a FPR=n/a

Paired A->B on gold positives (broad flag): {"flagged_A_only": 0, "flagged_B_only": 1, "flagged_both": 0, "flagged_neither": 3}

Positive label types (response-span annotations): {"Evident Baseless Info": 13, "Evident Conflict": 6}
