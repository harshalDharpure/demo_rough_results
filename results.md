# Research overview and results

**Task:** Given a complete crash video, write an explanation. Each claim must cite a time interval, or the model must abstain.

**Dataset:** Crash-1500. Train 1048 / val 224 / test 226. Gold Crash-EC labels on 175 test videos.

**Latest NLI (current method, 226 test videos):** **17.7%** on A5 (full Method 4).

---

## Method 4 — what we did

Method 4 is a two-stage pipeline. We did **not** rebuild VideoLLaMA3. We used the published model for Stage 1 and added our check in Stage 2.

**Stage 1 — draft.** The crash video goes into VideoLLaMA3. It writes a free-text explanation of what happened.

**Stage 2 — Temporal Evidence Contract (TEC).** That draft is split into short claims. Each claim must look like: text + time interval in the video + how sure we are (observed / derived / hypothesised / undetermined). A verifier then looks at the cited frames. If the frames support the claim, we keep it. If not, we revise it or mark it undetermined.

We ran this as an ablation ladder (add one piece at a time) on the full test set:

| ID | What we ran |
|----|-------------|
| A0 | Raw VideoLLaMA3 draft only |
| A1 | Same, but more frames |
| A2 | Timed prompt: ask the model for claims **and** timestamps |
| A3 | Add verifier (check claims vs cited frames; no rewrite) |
| A4 | Add revision / abstain when the check fails |
| A5 | Full Method 4: all of the above, then drop unsupported claims from the final report |

We also ran a weak baseline (Method 1: caption each frame, then merge) and LLaVA-NeXT sparse as Stage-1 comparisons. Qwen2.5-VL was planned and not finished.

For A5 we tested whether the model actually uses the frames it cites: remove those frames (confidence should drop = ERS) and remove unrelated frames (confidence should stay = IRS).

---

## NLI (latest)

| Method | NLI |
|--------|----:|
| A0 raw | 2.7% |
| A1 more frames | 5.8% |
| A2 timed prompt | 17.3% |
| A3 verifier | 17.3% |
| A4 revision | 17.7% |
| **A5 full Method 4** | **17.7%** |

Older LLaVA-NeXT (not Method 4): archived zero-shot 66.4% (not locked; live file 3.1%); fine-tuned 34.1%.

---

## Gold evidence (175 videos)

| Method | EC-F1 | UCCR ↓ |
|--------|------:|-------:|
| A2 timed prompt | **0.654** | 0.034 |
| A3 verifier | 0.635 | 0.034 |
| A4 revision | 0.635 | 0.034 |
| A5 full Method 4 | 0.604 | **0.029** |

---

## Ablation ladder (226 videos)

| Method | EC-F1 | UCCR ↓ |
|--------|------:|-------:|
| A0 raw | 0.680 | 0.047 |
| A1 more frames | 0.680 | 0.011 |
| A2 timed prompt | 0.654 | 0.034 |
| A3 verifier | 0.635 | 0.034 |
| A4 revision | 0.635 | 0.034 |
| A5 full Method 4 | 0.604 | 0.029 |

---

## Interventions (A5, 224 videos)

| Metric | Value |
|--------|------:|
| ERS ↑ | **0.694** |
| IRS ↑ | **0.996** |

---

## Short takeaway

NLI improved **2.7% → 17.7%**, mainly at A2 (timed prompt). A5 is more conservative (UCCR 0.029) and uses cited frames (ERS 0.694). Best gold EC-F1 is still A2 (**0.654**), not A5 (**0.604**).
