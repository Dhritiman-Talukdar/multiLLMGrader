That's actually a much stronger evaluation setup than before.

Since you now have **per-question grades from three independent human graders**, you should not simply compare each LLM against a single human. Instead, treat the human graders as the reference standard and first quantify human agreement.

I'd structure the evaluation like this.

---

# 1. Measure Human-Human Agreement (Upper Bound)

Before judging any LLM, determine how consistent the humans are.

Compute for every question:

* Inter-rater reliability

  * ICC (Intraclass Correlation Coefficient) ← best for numeric grades
  * Krippendorff's Alpha
  * Weighted Cohen's Kappa (pairwise)
  * Kendall's W (optional)

Example

| Question | Human A | Human B | Human C |
| -------- | ------- | ------- | ------- |
| Q1       | 8       | 9       | 8       |
| Q2       | 4       | 5       | 5       |

If the humans only agree at ICC = **0.82**, then expecting an LLM to achieve 0.99 is unrealistic.

Human agreement becomes the ceiling.

---

# 2. Create a Reference Grade

Instead of choosing one grader, create a consensus.

Possible methods

### Mean

[
Reference=\frac{G_1+G_2+G_3}{3}
]

Best for continuous marks.

---

### Median

Better if one grader is unusually harsh.

---

### Majority (for discrete scores)

If grades are integers,

8,8,9 → 8

---

# 3. Compare Every LLM Against Consensus

For every model compute

### Mean Absolute Error (MAE)

Average grading error.

Lower is better.

---

### RMSE

Penalizes large mistakes.

---

### Pearson Correlation

Measures linear agreement.

---

### Spearman Correlation

Measures ranking agreement.

---

### R²

Explains grading variance.

---

### Bias

Average

LLM − Human

Positive

→ generous grader

Negative

→ harsh grader

---

### Standard Deviation of Error

Shows consistency.

Two models may have MAE=1

One always differs by ±1

Another differs by ±4 sometimes.

Second model is less reliable.

---

# 4. Does the LLM Perform Like a Human?

This is the interesting analysis.

Treat the LLM as a **fourth grader**.

Compute

ICC

for

Human1

Human2

Human3

LLM

If ICC barely drops,

then the LLM behaves like another human examiner.

This is much stronger evidence than simply reporting MAE.

---

# 5. Compare Against Individual Humans

Compute

MAE(LLM, Human1)

MAE(LLM, Human2)

MAE(LLM, Human3)

Then compare to

MAE(Human1, Human2)

MAE(Human1, Human3)

MAE(Human2, Human3)

If

LLM↔Human ≈ Human↔Human

then the LLM is effectively within human variability.

This is often the strongest practical argument.

---

# 6. Per-Question Analysis

Some questions are harder.

For every question compute

* Human variance
* LLM variance
* Average LLM error

Then identify

Easy questions

* Humans agree
* LLM agrees

Hard questions

* Humans disagree
* LLM also struggles

Interesting failures

* Humans agree
* LLM fails badly

Those are the questions worth investigating qualitatively.

---

# 7. Statistical Significance

When comparing two LLMs,

don't rely only on average MAE.

Use paired statistical tests such as:

* Wilcoxon signed-rank test (recommended if error distributions are non-normal)
* Paired t-test (if assumptions hold)

This tells you whether one model is genuinely better rather than just slightly lower in average error.

---

# 8. Visualizations

I would include:

* Scatter plot: Human consensus vs LLM score
* Bland–Altman plot (agreement analysis)
* Boxplots of grading error by model
* Heatmap of absolute error (question × model)
* Correlation matrix (all graders + all LLMs)
* Distribution of grading bias
* Violin plots of errors
* Radar chart comparing metrics (MAE, RMSE, ICC, Bias, etc.)

---

# 9. Overall Ranking

For each LLM, report:

| Metric                  | Model A | Model B | Model C |
| ----------------------- | ------- | ------- | ------- |
| MAE ↓                   | ✓       |         |         |
| RMSE ↓                  |         | ✓       |         |
| Pearson ↑               |         |         |         |
| Spearman ↑              |         |         |         |
| ICC ↑                   |         |         |         |
| Bias ↓                  |         |         |         |
| Human Agreement Ratio ↑ |         |         |         |

You can also derive a composite score if you want a single ranking, but it's best to present the underlying metrics alongside it.

## Recommendation

Given your dataset with **three human graders and multiple LLMs**, I would build the evaluation around the question:

> **"Does the LLM behave like an additional qualified human grader?"**

To answer that convincingly, the core analyses should be:

1. **Human–human agreement** (ICC, Krippendorff's Alpha, pairwise agreement).
2. **Consensus human grade** (mean or median).
3. **LLM vs. consensus** (MAE, RMSE, Pearson, Spearman, bias).
4. **LLM vs. each human** compared with **human vs. human** agreement.
5. **ICC including the LLM as a fourth grader** to see whether adding the LLM preserves inter-rater reliability.
6. **Per-question error analysis** to identify where models succeed or fail.

This methodology is considerably more rigorous than comparing LLM scores to a single human grader and aligns well with evaluation practices used in educational assessment and automated grading research.
