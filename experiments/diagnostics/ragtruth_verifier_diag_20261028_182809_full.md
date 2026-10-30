# RAGTruth verifier diagnostic — ragtruth_verifier_diag_20261028_182809_full

Claims: 17 gold-hallucinated + 60 gold-unflagged (seeded sample). Diagnostic prevalence is not natural prevalence: compare TPR/FPR, read precision/F1 as conditional on this mix.

## Flag = broad

| Condition | TP | FN | FP | TN | TPR [Wilson 95%] | FPR [Wilson 95%] | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| A_original | 0 | 17 | 6 | 54 | 0.000 [0.000, 0.184] | 0.100 [0.047, 0.201] | 0.000 | 0.000 |
| B_readable | 2 | 15 | 8 | 52 | 0.118 [0.033, 0.343] | 0.133 [0.069, 0.242] | 0.200 | 0.148 |

By task (broad):

| Condition | Task | TP/FN | FP/TN | TPR | FPR |
|---|---|---|---|---|---|
| A_original | Data2txt | 0/16 | 3/17 | 0.000 | 0.150 |
| A_original | QA | 0/0 | 3/17 | n/a | 0.150 |
| A_original | Summary | 0/1 | 0/20 | 0.000 | 0.000 |
| B_readable | Data2txt | 2/14 | 1/19 | 0.125 | 0.050 |
| B_readable | QA | 0/0 | 6/14 | n/a | 0.300 |
| B_readable | Summary | 0/1 | 1/19 | 0.000 | 0.050 |

## Flag = strict

| Condition | TP | FN | FP | TN | TPR [Wilson 95%] | FPR [Wilson 95%] | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| A_original | 0 | 17 | 4 | 56 | 0.000 [0.000, 0.184] | 0.067 [0.026, 0.159] | 0.000 | 0.000 |
| B_readable | 2 | 15 | 7 | 53 | 0.118 [0.033, 0.343] | 0.117 [0.058, 0.222] | 0.222 | 0.154 |

By task (strict):

| Condition | Task | TP/FN | FP/TN | TPR | FPR |
|---|---|---|---|---|---|
| A_original | Data2txt | 0/16 | 2/18 | 0.000 | 0.100 |
| A_original | QA | 0/0 | 2/18 | n/a | 0.100 |
| A_original | Summary | 0/1 | 0/20 | 0.000 | 0.000 |
| B_readable | Data2txt | 2/14 | 1/19 | 0.125 | 0.050 |
| B_readable | QA | 0/0 | 5/15 | n/a | 0.250 |
| B_readable | Summary | 0/1 | 1/19 | 0.000 | 0.050 |

## Label counts

- A_original: {"HALLUCINATED": {"SUPPORTED": 17}, "UNFLAGGED": {"SUPPORTED": 54, "INSUFFICIENT": 2, "CONFLICTED": 4}}
- B_readable: {"HALLUCINATED": {"SUPPORTED": 15, "CONFLICTED": 2}, "UNFLAGGED": {"SUPPORTED": 52, "CONFLICTED": 7, "INSUFFICIENT": 1}}

## Latency (recorded per-call, seconds) and evidence size

- A_original: mean recorded latency 15.363s over 77 calls; mean evidence chars 1930
- B_readable: mean recorded latency 13.276s over 77 calls; mean evidence chars 1936

## Condition B split by whether the whole source was visible

- full_source_visible=True: n=64 TP=2 FN=13 FP=7 TN=42 TPR=0.133 FPR=0.143
- full_source_visible=False: n=13 TP=0 FN=2 FP=1 TN=10 TPR=0.000 FPR=0.091

Paired A->B on gold positives (broad flag): {"flagged_A_only": 0, "flagged_B_only": 2, "flagged_both": 0, "flagged_neither": 15}

Positive label types (response-span annotations): {"Evident Baseless Info": 13, "Evident Conflict": 6}
