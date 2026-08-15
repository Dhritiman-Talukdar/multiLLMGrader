# Design: Journal Article — Multi-LLM vs. Multi-Human Grading

**Date:** 2026-08-15
**Status:** Approved, drafting
**Target venue:** Engineering education journal (IEEE Transactions on Education / Computers & Education class)
**Output format:** Markdown manuscript (`docs/paper/manuscript.md`), pandoc-convertible

---

## 1. Working Title

> **Human Agreement Is the Ceiling: Evaluating Seven Frontier LLMs as Rubric Graders Across Two Engineering Courses**

## 2. Contribution Statement

Three layered contributions:

1. **Method.** A human-ceiling evaluation protocol for LLM graders: measure inter-human
   reliability *first*, form a consensus reference, then test whether the LLM survives
   insertion as an additional rater. Demonstrated empirically to change conclusions —
   not merely asserted.
2. **Evidence.** Seven frontier models x 3 runs x 2 dissimilar engineering courses
   (945 model-gradings total) against three independent human graders per course.
3. **Guidance.** No model reaches human variability in either domain, but the dominant
   error component is a *systematic harshness offset* rather than random noise. That is
   correctable, and it licenses specific assistive (not autonomous) deployments.

## 3. Key Empirical Results

| Quantity | OS (primary) | Biomaterials (contrast) |
|---|---|---|
| Students / question-instances | 40 / 240 | 5 / 25 |
| Rubric total | 133 pts, 6 questions | 10 pts, 5 questions |
| Input modality | structured JSON answers | direct PDF ingestion |
| Human ceiling ICC(2,1) | 0.956 | 0.504 |
| Human Krippendorff alpha | 0.956 | 0.469 |
| Human-human MAE | 1.113 pts/q | 0.227 pts/q |
| Best model (composite) | Claude Opus (0.978) | Claude Opus (0.859) |
| Worst model | GPT-4o (0.097) | GPT-4o (0.402) |
| Best MAE / ceiling ratio | 2.80x | 1.76x (Haiku); 1.80x (Opus) |
| ICC drop as 4th rater | -0.091 .. -0.209 | -0.077 .. -0.161 |
| Models within human variability | 0 / 7 | 0 / 7 |
| Models significantly harsh | 5 / 7 (n=40) | 7 / 7 (n=25) |

**Cross-domain replication — stated precisely.** Checked against the data rather than
assumed. The full ranking does *not* replicate: Spearman rho between the two courses is
0.750 (p = 0.052) on the composite, 0.571 (p = 0.18) on MAE, 0.643 (p = 0.12) on
ICC-as-4th-rater, all at n = 7 models. The paper must therefore claim only what holds:

- **The top replicates exactly.** Claude Opus is rank 1 and Claude Sonnet rank 2 in both
  courses, on every individual metric.
- **The OpenAI models are bottom-tier in both**, though they swap places (GPT-4o last on
  OS, GPT-5 last on Biomaterials).
- **The middle does not order consistently** — Gemini 2.5 Pro is 3rd on OS but 5th on
  Biomaterials; Claude Haiku is 6th on OS but 4th on Biomaterials.
- **What replicates without qualification** are the two verdicts: 0/7 models within human
  variability, and every model harsh, in both courses.

Do not write "the ranking replicates". Write that the extremes replicate and the middle
does not, and report the rank correlations with their p-values.

**The hinge result (§5.7):** naive single-reference + pseudoreplicated analysis flips the
sign of measured bias for **5 of 7** models on OS (Sonnet +2.35 -> -6.46; GPT-5
+3.39 -> -5.42; Opus +4.50 -> -4.31; Flash +6.35 -> -2.47; Gemini Pro +6.76 -> -2.06),
reports p < 0.001 for the two models the corrected analysis finds *not* significantly
biased (Gemini Pro, Flash), and reports p = 0.667 for GPT-4o, which is in fact
significantly harsh (corrected p = 0.0013). Root cause: TA2 never graded Q6, so `ta2_total_score` is a
Q1-Q5 sum and the naive `ta_avg` reference is ~8.8 pts too low; runs were also pooled as
independent observations (n=120) rather than averaged to the student (n=40).

## 4. Manuscript Structure

| Section | Core claim | Evidence source |
|---|---|---|
| 1. Introduction | Single-reference evaluation is the field default and it overstates LLM graders | — |
| 2. Related work | Autograding, LLM-as-judge, IRR in education | `[CITE: ...]` placeholders only |
| 3. Human-ceiling framework | The protocol: IRR -> consensus -> vs-consensus metrics -> 4th-rater insertion -> within-human test -> run-to-run noise | Methods prose |
| 4. System & study design | Pipeline, models, runs, courses, rubrics, missing-data handling | `grading_service.py`, driver scripts |
| 5. Study 1: OS (primary) | 5.1 naive -> 5.2 ceiling -> 5.3 vs consensus -> 5.4 4th rater -> 5.5 within-human -> 5.6 noise -> **5.7 what changed** | `llm_vs_human_grading_analysis.ipynb` Phase 9 |
| 6. Study 2: Biomaterials (contrast) | Low human ceiling = a different bar, same verdict, same ordering | `multiLLM_multiHuman_analysis.ipynb` |
| 7. Cross-domain synthesis | Extremes replicate, middle does not (rho = 0.75, p = 0.052); ceiling ratio differs; harshness universal (12/14 significant) | new table + Fig. 9 |
| 8. Discussion | Calibration offset; assistive deployment modes; accuracy/latency tradeoff | latency CSVs |
| 9. Threats to validity | See §6 below | — |
| 10. Conclusion | — | — |

Biomaterials is deliberately **not** co-equal. Its n=5 arm cannot carry independent weight
(MAE CIs overlap heavily: Claude Opus [0.28, 0.49] vs GPT-5 [0.40, 0.70]). It earns its
place because its *low* human ceiling makes the framework's verdict-changing property
visible: against a single strict grader the models look worse than they are, and against a
0.504-ICC panel the appropriate bar is manifestly not "ICC > 0.9".

## 5. Figures and Tables

**Figures (8 main text).** Reused from existing `outputs/`:

Numbered in document order (11 figures). Reused from existing `outputs/`:

| Fig | Section | Source |
|---|---|---|
| 1 | 5.3 | OS `4_2_llm_vs_ta_scatter.png` |
| 2 | 5.3 | OS `5_4_bland_altman.png` |
| 4 | 5.5 | OS `9_5_within_human_variability.png` |
| 5 | 5.6 | OS `8_2_per_question_mae_heatmap.png` |
| 6 | 5.6 | OS `9_6_run_to_run_noise.png` |
| 7 | 6.2 | Bio `04_bias_by_model.png` |
| 8 | 6.2 | Bio `10g_radar.png` |
| 11 | 7.4 | OS `4_6_latency_boxplot.png` |

**Generated** by `docs/paper/make_paper_figures.py` into `docs/paper/figures/`:

- **Fig. 3** (§5.4) — `fig3_icc_as_fourth_rater_os.png`. Insertion test, Operating Systems.
- **Fig. 9** (§6.3) — `fig9_icc_as_fourth_rater_bio.png`. Insertion test, Biomaterials.
- **Fig. 10** (§7.2) — `fig10_cross_domain_ceiling_ratio.png`. Cross-domain MAE-to-ceiling
  ratio, carrying §7's replication claim.

**The insertion test gets one figure per course, not one shared two-panel figure.** A
combined figure is cited from both §5.4 and §6.3, so it would print the Biomaterials result
inside the Operating Systems results section and vice versa. Each course's figure is scaled
to its own human baseline (`xlim = baseline * 1.18`) so the dashed ceiling line sits in a
comparable position across the two, and each bar is annotated with its percentage of that
course's ceiling — which is what makes the two comparable despite the very different ICC
ranges (0.747–0.865 vs 0.342–0.426). Value labels sit *inside* the bars; placed outside they
run across the baseline line.

**Tables (6).** Study design; human ceiling both datasets; OS metrics vs consensus;
naive vs corrected; cross-domain ranking; latency.

## 6. Threats to Validity (must be reported)

Discovered while reading the implementation — these are real confounds, not boilerplate:

1. **Prompt is not identical across providers.** `_build_bulk_prompt` emits a compact
   delimiter format for Gemini with an explicit "under 30 words per field" instruction,
   and a JSON format for OpenAI/Anthropic. Gemini's results are therefore not a
   clean model comparison; the prompt is a confound.
2. **Decoding settings differ.** Temperature 0.1 for all models except GPT-5, which uses
   `reasoning_effort="high"` with no temperature control. GPT-5's run-to-run SD is not
   measured under the same conditions as the others.
3. **Token budgets differ.** `max_tokens` 2000 for Anthropic/Gemini vs 16384 for
   non-reasoning OpenAI models, which can truncate long rubric feedback asymmetrically.
4. **Input modality differs between studies.** OS grades structured JSON text answers via
   `grade_submission`; Biomaterials grades PDFs directly via `grade_pdf_direct` (Bedrock
   document blocks with citations enabled for Anthropic). Cross-study differences are
   confounded with modality.
5. **Biomaterials is underpowered** (5 students, 25 question-instances).
6. **Consensus-as-truth.** The mean of three humans is a reference, not ground truth;
   on Biomaterials the humans themselves agree only moderately (ICC 0.504).
7. **Range restriction on Biomaterials** — 5 questions x 2 pts, most human scores in
   [1.5, 2.0], which mechanically depresses ICC and inflates the apparent LLM gap.
8. **Single assignment per course**, single institution, one rubric each.
9. **Model versions are a moving target**; results are a snapshot of the tested versions.

## 7. Scope Boundaries

**In scope:** full IMRaD draft in Markdown; two new figures; tables generated from
existing CSV/notebook outputs.

**Out of scope:** re-running any grading; new statistical analyses beyond assembling
what the notebooks already computed; real literature citations (placeholders only);
per-model API cost figures (not available in the data).
