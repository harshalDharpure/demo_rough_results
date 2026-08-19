# Research overview and results

**Task:** Given a complete crash video, write an explanation. Each claim must point to a time interval, or the model must say it is uncertain.

**Current method:** Method 4 — Video Large Language Model Meta Artificial Intelligence 3 writes a draft; Temporal Evidence Contracts check or revise each claim.

**Dataset:** Crash-1500. Train 1048 / validation 224 / test 226. Gold evidence labels on 175 test videos.

**Latest Natural Language Inference (current method, 226 test videos):** **17.7%** on Ablation 5 (full Method 4).

---

## Natural Language Inference (latest)

| Method | Natural Language Inference |
|--------|---------------------------:|
| Ablation 0 — raw Video Large Language Model | 2.7% |
| Ablation 1 — more frames | 5.8% |
| Ablation 2 — timed prompt | 17.3% |
| Ablation 3 — verifier | 17.3% |
| Ablation 4 — revision | 17.7% |
| **Ablation 5 — full Method 4** | **17.7%** |

Older Large Language and Vision Assistant — Next (not the current method): archived zero-shot 66.4% (not locked; live file is 3.1%); fine-tuned 34.1%.

---

## Gold evidence scores (175 videos)

| Method | Evidence Contract F1-score | Unsupported Causal Claim Rate (lower better) |
|--------|---------------------------:|---------------------------------------------:|
| Ablation 2 — timed prompt | **0.654** | 0.034 |
| Ablation 3 — verifier | 0.635 | 0.034 |
| Ablation 4 — revision | 0.635 | 0.034 |
| Ablation 5 — full Method 4 | 0.604 | **0.029** |

---

## Ablation ladder (226 videos)

| Method | Evidence Contract F1-score | Unsupported Causal Claim Rate |
|--------|---------------------------:|------------------------------:|
| Ablation 0 — raw | 0.680 | 0.047 |
| Ablation 1 — more frames | 0.680 | 0.011 |
| Ablation 2 — timed prompt | 0.654 | 0.034 |
| Ablation 3 — verifier | 0.635 | 0.034 |
| Ablation 4 — revision | 0.635 | 0.034 |
| Ablation 5 — full Method 4 | 0.604 | 0.029 |

---

## Interventions (Ablation 5, 224 videos)

| Metric | Value |
|--------|------:|
| Evidence Removal Sensitivity (higher better) | **0.694** |
| Irrelevant Removal Stability (higher better) | **0.996** |

---

## Short takeaway

Natural Language Inference improved **2.7% → 17.7%** (mainly at the timed prompt). Full Method 4 is more conservative (unsupported causal claims 0.029) and uses the frames it cites (0.694). Gold Evidence Contract F1-score is still better at Ablation 2 (**0.654**) than Ablation 5 (**0.604**).
