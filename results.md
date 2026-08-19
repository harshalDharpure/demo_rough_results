# Complete Research Report

**Project:** Vision-Language Model Road Crash Understanding  
**Working name:** CrashGraph  
**Dataset:** Crash-1500 (one thousand five hundred dashcam crash videos)  
**Report date:** 19 August 2026  
**Written in simple language.** Every short name is also given in full form the first time it appears.

This file is the full status report: what we did, what the numbers mean, the latest Natural Language Inference scores, which method they belong to, and where we are actually improving.

---

## 1. The project in one paragraph

We take a complete road-crash video and ask a computer to write an explanation of what happened. The explanation must be faithful to the video. That means: if the model says “the black car ran a red light,” those frames must actually show that. If the video does not show it, the model should say it is uncertain, not invent a cause.

We are **not** trying to predict a crash before it happens. Accident anticipation (guessing that a crash will occur in the next few seconds) was dropped after mentor feedback. The task is: **the crash has already happened; explain it honestly.**

---

## 2. Dictionary — full forms of every short name

| Short name | Full form | Simple meaning |
|------------|-----------|----------------|
| VLM | Vision-Language Model | A model that looks at images or video and writes text |
| LLM | Large Language Model | A text model (the writing part of a Vision-Language Model) |
| LLaVA-NeXT | Large Language and Vision Assistant — Next | An image-and-text model we used first |
| VideoLLaMA3 | Video Large Language Model Meta Artificial Intelligence 3 | A **video** model (our current main generator) |
| Qwen | Tongyi Qianwen (Alibaba open model family) | Another Vision-Language Model; planned as a second backbone; **not finished** |
| CLIP | Contrastive Language-Image Pre-training | A cheap image–text matcher; **weak checker only**, not our main method |
| LoRA | Low-Rank Adaptation | A cheap way to fine-tune a large model |
| QLoRA | Quantized Low-Rank Adaptation | Low-Rank Adaptation with compressed (quantized) weights |
| NLI | Natural Language Inference | A test: does sentence B follow from sentence A? |
| BLEU | Bilingual Evaluation Understudy | Word-overlap score vs a human caption |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation | Another word-overlap score (longest matching phrases) |
| METEOR | Metric for Evaluation of Translation with Explicit Ordering | Overlap score that also allows synonyms |
| BERTScore | Bidirectional Encoder Representations from Transformers Score | Meaning similarity using embeddings |
| CIDEr | Consensus-based Image Description Evaluation | Caption score used in image captioning papers |
| SPICE | Semantic Propositional Image Caption Evaluation | Scene-graph style caption score |
| GT | Ground Truth | The human-written Excel explanation |
| TEC | Temporal Evidence Contract | Each claim must point to a time interval in the video, or abstain |
| Crash-EC | Crash Evidence Contracts | Our gold labels: claims + time intervals on 175 test videos |
| CEP | Claim Evidence Precision | Of the claims the model wrote, how many are supported |
| CER | Claim Evidence Recall | Of the gold claims, how many the model recovered |
| EC-F1 | Evidence Contract F1-score | Harmonic mean of Claim Evidence Precision and Claim Evidence Recall |
| UCCR | Unsupported Causal Claim Rate | Fraction of cause-claims that are **not** supported (lower is better) |
| RCS | Report Consistency Score | Internal contradiction score; in our runs it is always 1.0, so it is useless |
| ERS | Evidence Removal Sensitivity | Confidence should **drop** when we delete the frames the claim cites (higher is better) |
| IRS | Irrelevant Removal Stability | Confidence should **stay** when we delete unrelated frames (higher is better) |
| HAG | Hindsight Attribution Gap | Does the model invent a pre-crash cause only after seeing the wreck? |
| TE-IoU | Temporal Evidence Intersection over Union | How well the predicted time interval overlaps the gold interval |
| IAA | Inter-Annotator Agreement | Do two people label the same way? Ours is still a computer stand-in |
| HAG / ERS / IRS “q” | Confidence / support score | How sure the model (or verifier) is about a claim |
| CCD | Car Crash Dataset | Original source family of many of these clips (Bao and others, Association for Computing Machinery Multimedia 2020) |
| GPU | Graphics Processing Unit | The video card used to run the models |
| NeurIPS | Conference on Neural Information Processing Systems | Target conference |
| Q1 | First-quartile journal | Top-band journal target |

**Natural Language Inference (the metric you asked about) in plain words**

We use a model named **Robustly Optimized Bidirectional Encoder Representations from Transformers, large, Multi-Genre Natural Language Inference** (RoBERTa-large-MNLI).

- **Premise** = the Excel human explanation  
- **Hypothesis** = the model’s generated text  

**Natural Language Inference Entailment Accuracy** = the share of videos where that checker says the generated text is entailed by (agrees with) the human explanation.

Important: this checker **never sees the video**. A high Natural Language Inference score means “the text looks like the Excel caption,” not “every fact is visible in the video.” That is why the new paper metrics are Evidence Contract F1-score, Unsupported Causal Claim Rate, Evidence Removal Sensitivity, and Irrelevant Removal Stability.

---

## 3. Dataset and split

| Item | Detail |
|------|--------|
| Videos | About 1500 files in `video1500/` |
| Ground truth workbook | `Car_Crash_Text_Dataset_ground_truth.xlsx` |
| Main text field | Explanation (the human paragraph) |
| Other fields | Severity of the Crash, Type of Vehicles involved, Number of Vehicles, Location of impact, Start/End of Crash, Ambiguity, Camera View, Weather Conditions |
| Usable explanations | About 1498 videos |
| Split (random seed 42) | Train 1048 / Validation 224 / **Test 226** |
| Gold Crash Evidence Contracts | **175 test videos** with atomic claims and time intervals |

---

## 4. What we did, in order

### Phase 1 — First pipeline (May 2026): Large Language and Vision Assistant — Next

We built a full research pipeline:

1. Validate videos and parse the Excel file.  
2. Split train / validation / test.  
3. Sample frames: dense (every frame), every 3rd, every 5th, every 10th.  
4. Run **zero-shot** Large Language and Vision Assistant — Next (no extra training).  
5. Fine-tune with Low-Rank Adaptation for 5 epochs.  
6. Score with Bilingual Evaluation Understudy, Recall-Oriented Understudy for Gisting Evaluation, Metric for Evaluation of Translation with Explicit Ordering, Bidirectional Encoder Representations from Transformers Score, Consensus-based Image Description Evaluation, Semantic Propositional Image Caption Evaluation, and Natural Language Inference.  
7. Try 20 combinations of sampling × prompt on all 226 test videos.  
8. Try a Natural Language Inference “optimized” filter.

**What this phase actually did (honest):** unless collage mode was on, the model usually received **one middle frame**, not a true video. So this phase is better described as **single-frame crash captioning**, not sparse-video understanding. We keep it as a **weak baseline**, not as the main paper claim.

### Phase 2 — Redesign (July–August 2026): CrashGraph / Method 4

After an audit and mentor feedback we locked one problem:

> Complete crash video → text explanation where every claim is tied to a time interval, or the model abstains.

**Method 4** has two stages:

**Stage 1 — draft (from existing video papers, we do not reimplement the encoder)**

```text
Crash video → Video encoder → Temporal connector → Large Language Model → Draft summary
```

Main Stage-1 model: **Video Large Language Model Meta Artificial Intelligence 3**.

**Stage 2 — our addition: Temporal Evidence Contracts**

```text
Draft → split into claims + timestamps → check against frames → keep, revise, or abstain
```

**Ablation ladder** (add one piece at a time to see what helps):

| Code | Full name | What we add |
|------|-----------|-------------|
| A0 | Ablation 0 — raw | Video Large Language Model only |
| A1 | Ablation 1 — more frames | More sampled frames |
| A2 | Ablation 2 — timed prompt | Ask for claims **and** time intervals |
| A3 | Ablation 3 — verifier | Check each claim against cited frames (no rewrite) |
| A4 | Ablation 4 — revision | Rewrite or mark undetermined if the check fails |
| A5 | Ablation 5 — full Method 4 | All of the above, and drop unsupported claims from the final report |

We also built **Method 1** as a weak baseline: caption each frame separately, then merge the captions with a text model.

Gold Crash Evidence Contracts (175 videos), intervention tests (delete cited frames / keep only cited frames / shuffle), and frozen paper tables (9 August 2026; cleaned 15 August 2026) are done.

---

## 5. Latest Natural Language Inference scores

Natural Language Inference Entailment Accuracy on the **full test set of 226 videos**, unless noted.

### 5.1 Current research method — Method 4 (Video Large Language Model Meta Artificial Intelligence 3)

These are the **latest Natural Language Inference numbers for the method we are developing now.**

| Method (full name) | Natural Language Inference Entailment Accuracy | Percent |
|--------------------|-----------------------------------------------:|--------:|
| Ablation 0 — raw Video Large Language Model | 0.027 | **2.7%** |
| Ablation 1 — more frames | 0.058 | **5.8%** |
| Ablation 2 — timed prompt | 0.173 | **17.3%** |
| Ablation 3 — verifier (no rewrite) | 0.173 | **17.3%** |
| Ablation 4 — revision | 0.177 | **17.7%** |
| **Ablation 5 — full Method 4** | **0.177** | **17.7%** |

**Latest Method 4 Natural Language Inference: 17.7% on Ablation 4 and Ablation 5 (full Method 4).**

Compared with Ablation 0 (2.7%), timed prompting plus verify/revise **raises** Natural Language Inference from 2.7% to 17.7%. Ablation 3 and Ablation 4/5 are almost the same, so **revision is not yet changing Natural Language Inference**.

Contrastive Language-Image Pre-training (weak checker, **not** the main method) reached about **31.4%** Natural Language Inference on Ablation 5. Do not present that as Method 4. It is a cheap verifier ablation.

### 5.2 Older Large Language and Vision Assistant — Next track (legacy)

| Method (full name) | Natural Language Inference Entailment Accuracy | Percent | Can we trust it? |
|--------------------|-----------------------------------------------:|--------:|------------------|
| Zero-shot, every 5th frame, structured-event prompt (archived log) | 0.664 | **66.4%** | **No — restored from an old log.** The live prediction file for the same folder now scores **3.1%**. Do not show 66.4% as a locked result until it is re-run. |
| Same setting, current `metrics.json` file | 0.031 | **3.1%** | This is what is on disk now for that folder |
| Zero-shot collage (several frames in one image) | 0.044 | **4.4%** | Yes, as a negative result |
| Fine-tuned Low-Rank Adaptation, best checkpoint | 0.341 | **34.1%** | Yes as a lexical-vs-faithfulness contrast |
| Fine-tuned, epoch 2, collage | 0.310 | **31.0%** | Yes |
| Fine-tuned version 2, collage, faithfulness prompt | 0.058 | **5.8%** | Yes |
| Natural Language Inference–optimized filter | 0.602 | **60.2%** | **No for a test score** — it used the **test Excel explanation** to keep or drop sentences |

### 5.3 Which Natural Language Inference number should you say out loud?

| If someone asks… | Answer |
|------------------|--------|
| Latest score on **our current method** | **17.7% Natural Language Inference Entailment Accuracy on full Method 4 (Ablation 5), 226 test videos** |
| Highest **Method 4** Natural Language Inference | Same: **17.7%** (Ablation 4 / Ablation 5) |
| Highest **legacy** number on disk | 66.4% archived, but **not reproducible** from current predictions |
| Fine-tuning | Improves Bilingual Evaluation Understudy / Recall-Oriented Understudy for Gisting Evaluation, **hurts** Natural Language Inference (34.1% vs the archived 66.4%) |

---

## 6. The paper metrics (more important than Natural Language Inference now)

Gold Crash Evidence Contracts, **175 videos** (frozen tables 9 August 2026).

| Method | Evidence Contract F1-score | Temporal Evidence Intersection over Union | Epistemic status F1-score | Unsupported Causal Claim Rate (lower better) |
|--------|---------------------------:|------------------------------------------:|--------------------------:|---------------------------------------------:|
| Ablation 2 — timed prompt | **0.654** | 0.322 | 0.181 | 0.034 |
| Ablation 3 — verifier | 0.635 | 0.322 | 0.183 | 0.034 |
| Ablation 4 — revision | 0.635 | 0.322 | 0.183 | 0.034 |
| Ablation 5 — full Method 4 | 0.604 | 0.297 | **0.212** | **0.029** |

Full ladder on 226 videos (Evidence Contract F1-score / Unsupported Causal Claim Rate):

| Method | Evidence Contract F1-score | Unsupported Causal Claim Rate |
|--------|---------------------------:|------------------------------:|
| Ablation 0 — raw | 0.680 | 0.047 |
| Ablation 1 — more frames | 0.680 | 0.011 |
| Ablation 2 — timed prompt | 0.654 | 0.034 |
| Ablation 3 — verifier | 0.635 | 0.034 |
| Ablation 4 — revision | 0.635 | 0.034 |
| Ablation 5 — full Method 4 | 0.604 | 0.029 |

**Interventions on Ablation 5** (Video Large Language Model as scorer, 224 videos):

| Metric | Full form | Value | Meaning |
|--------|-----------|------:|---------|
| ERS | Evidence Removal Sensitivity | **0.694** | When we delete the cited frames, confidence **drops** — the model is using those frames |
| IRS | Irrelevant Removal Stability | **0.996** | When we delete unrelated frames, confidence **stays** |
| HAG | Hindsight Attribution Gap | 0.064 | Small; computed on **only 11 claims** — do not lead with this |

Baselines (Evidence Contract F1-score): Method 1 caption-and-merge **0.667** (226 videos); Large Language and Vision Assistant — Next sparse **0.401** (100 videos); Video Large Language Model timed draft **0.416** (226 videos). Tongyi Qianwen 2.5 Vision-Language: **0 videos** (disk was full).

---

## 7. Where we are improving (and where we are not)

### Improving

1. **Natural Language Inference on Method 4**  
   Ablation 0 **2.7%** → Ablation 2 **17.3%** → Ablation 5 **17.7%**.  
   The gain is almost all from the **timed prompt** (Ablation 2), not from the verifier.

2. **Unsupported Causal Claim Rate**  
   Ablation 5 **0.029** is better (lower) than Ablation 2 **0.034**. Full Method 4 makes **fewer unsupported cause statements**.

3. **Epistemic status F1-score**  
   Ablation 5 **0.212** is the best of A2–A5 (observed / derived / hypothesised / undetermined).

4. **Evidence use (the strongest Method 4 finding)**  
   Evidence Removal Sensitivity **0.694** and Irrelevant Removal Stability **0.996** on Ablation 5. This is what Temporal Evidence Contracts are for: the model should care about the frames it cites.

5. **Research story**  
   We moved from “every-fifth-frame captioning” to “claim + time interval or abstain.” That is a clearer paper.

### Not improving (say this honestly)

1. **Evidence Contract F1-score**  
   Ablation 0/1 **0.680** and Ablation 2 **0.654** beat Ablation 5 **0.604**. Full Method 4 is **not** the best gold-claim matcher. Do not call Ablation 5 “best overall.”

2. **Ablation 3 versus Ablation 4**  
   Same Evidence Contract F1-score **0.635** and same Natural Language Inference **17.3% vs 17.7%**. Revision is barely doing anything. The support threshold still needs a validation sweep so Ablation 4 is actually different from Ablation 3.

3. **Lexical overlap on Method 4**  
   Bilingual Evaluation Understudy-4 and Recall-Oriented Understudy for Gisting Evaluation-L stay around 0.03 / 0.23. Method 4 is not a BLEU-winning captioner. That is acceptable if we lead with evidence metrics.

4. **Legacy fine-tuning**  
   Low-Rank Adaptation **raises** Bilingual Evaluation Understudy-1 (**0.274 → 0.334**) and **lowers** Natural Language Inference (**archived 66.4% → 34.1%**). Word overlap up, agreement-with-Excel down.

5. **Second Vision-Language Model**  
   Tongyi Qianwen 2.5 Vision-Language is still **not run**.

6. **Human evaluation**  
   The 175-row sheet is **automatic proxy** from gold Temporal Evidence Contracts, not real people.

---

## 8. How to describe Method 4 in one honest sentence

Timed prompting (Ablation 2) is currently the **best gold Evidence Contract F1-score (0.654)**. Full Method 4 (Ablation 5) has **slightly worse gold matching (0.604)** but **fewer unsupported causal claims (0.029)** and a **large Evidence Removal Sensitivity (0.694)**. Latest Natural Language Inference on that full method is **17.7%**.

---

## 9. What is finished vs not finished

**Finished**

- Dataset processing and train / validation / test split  
- Large Language and Vision Assistant — Next zero-shot, Low-Rank Adaptation, 20-prompt grid  
- Method 4 code and Ablation 0–5 on 226 test videos  
- Gold Crash Evidence Contracts on 175 videos  
- Intervention tests for Ablation 5  
- Frozen, cleaned paper tables  

**Not finished (needed before a strong paper)**

- Real human ratings on about 50 videos, Ablation 2 versus Ablation 5, two raters  
- One second video model (Tongyi Qianwen 2.5 Vision-Language or similar)  
- Support-threshold sweep so Ablation 4 ≠ Ablation 3  
- Locked re-run of the 66.4% Natural Language Inference experiment, **or never report it**  
- Real two-person Inter-Annotator Agreement (current kappa is a bootstrap stand-in)  
- Causal-graph F1-score (no human causal edges yet)  
- The paper manuscript  

---

## 10. Numbers you should not put on a slide

| Number | Why not |
|--------|---------|
| 66.4% Natural Language Inference | Restored from a log; live file is 3.1% |
| 60.2% Natural Language Inference–optimized | Used the test explanation to filter sentences |
| Report Consistency Score = 1.000 | Always 1.0; dummy |
| Contrastive Language-Image Pre-training Claim Evidence Precision 0.895 | Cheap self-score, not gold |
| Self Claim Evidence Precision 0.821 | The model grading itself |
| Human useful ≈ 2.0 | Automatic proxy, not humans |
| Inter-Annotator Agreement kappa 0.95 | Computer copies, not two annotators |
| Hindsight Attribution Gap 0.064 as a headline | Only 11 claims |

---

## 11. Mentor-length summary

We use Crash-1500 (train 1048 / validation 224 / test 226). The first system was Large Language and Vision Assistant — Next captioning; it is a baseline only. The current system is Method 4: Video Large Language Model Meta Artificial Intelligence 3 writes a draft; Temporal Evidence Contracts check or revise each claim.

**Latest Natural Language Inference on the current method:** **17.7% entailment accuracy on Ablation 5 (full Method 4)**, up from **2.7%** on Ablation 0. The jump happens at the **timed prompt (Ablation 2, 17.3%)**.

**What is improving:** Natural Language Inference along Ablation 0 → Ablation 2 → Ablation 5; Unsupported Causal Claim Rate; Evidence Removal Sensitivity / Irrelevant Removal Stability.

**What is not:** gold Evidence Contract F1-score is still better at Ablation 2 (0.654) than Ablation 5 (0.604). Ablation 3 equals Ablation 4. Tongyi Qianwen and real human evaluation are still missing.

---

## 12. Where the files live

| What | Path |
|------|------|
| This report | `COMPLETE_RESEARCH_REPORT.md` |
| Clean paper tables | `results/method4/paper_q1/tables.md` |
| Method 4 Natural Language Inference (caption files) | `results/method4/metrics/A0_raw_caption.json` … `A5_full_method4_caption.json` |
| Archived 66.4% Natural Language Inference | `results/zero_shot/every_5th_structured_event_test/metrics_full_226.json` |
| Live 3.1% file for the same folder | `results/zero_shot/every_5th_structured_event_test/metrics.json` |
| Fine-tuned 34.1% Natural Language Inference | `results/finetuned/best_checkpoint/metrics_recomputed.json` |
| Gold Crash Evidence Contracts | `data/processed/crash_ec/gold_test175.jsonl` |
| Active plan | `CRASHGRAPH_NEURIPS_PLAN.md` |

---

*End of report. Numbers taken from disk on 19 August 2026. Method 4 Natural Language Inference is from the Ablation caption metric files. Gold Evidence Contract scores are from the frozen 9 August 2026 tables.*
