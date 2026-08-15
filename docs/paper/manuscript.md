# Human Agreement Is the Ceiling: Evaluating Seven Frontier LLMs as Rubric Graders Across Two Engineering Courses

**Authors.** [AUTHOR LIST]
**Affiliation.** [AFFILIATION]
**Corresponding author.** [EMAIL]

---

## Abstract

Large language models are increasingly proposed as graders for open-response
engineering coursework, but the published evidence is contradictory: reports range from
automated graders outperforming certified human re-graders to models falling far below
human inter-rater agreement on the same task. Most of this evidence is produced by
comparing a model against a *single* reference grader — a design that silently assumes
human graders agree with one another. We show that this assumption largely determines the
conclusion.

We evaluate seven frontier language models (GPT-4o, GPT-5, Gemini 2.5 Pro, Gemini 2.5
Flash, and Claude Haiku 4.5, Sonnet 4.6, and Opus 4.6), each run three times, on two
engineering assignments graded independently by three human graders each: a 133-point
Operating Systems assignment (40 students, 240 question instances) and a 10-point
Biomaterials assignment (5 students, 25 question instances). This yields 945 model
gradings against 755 human question-level judgements.

We propose and apply a **human-ceiling protocol**: quantify inter-human reliability first,
form a consensus reference from it, and then ask whether inserting the model as an
additional rater preserves the panel's reliability. Three findings follow. First, the
protocol is not cosmetic — on the Operating Systems assignment, a conventional
single-reference analysis reverses the *sign* of measured grading bias for five of seven
models and reports significant bias (p < 0.001) for two models that a corrected analysis
finds indistinguishable from the human consensus. Second, no model reaches human
variability in either course: the best model's disagreement with the human panel is 2.80x
the humans' disagreement with each other on Operating Systems and 1.76x on Biomaterials,
and inserting any model as a fourth rater lowers ICC(2,1) in both courses (−0.091 to
−0.209, and −0.077 to −0.161, respectively). Third, the error is not primarily random:
twelve of fourteen model-course pairs show statistically significant *harshness*, and the
offset accounts for most of the gap — which makes it correctable.

Because our two courses differ sharply in human ceiling (ICC 0.956 vs 0.504) while sharing
one protocol and one model set, we reproduce the literature's contradictory verdicts within
a single study, and argue that much of the field's disagreement concerns reference quality
rather than model capability.

We conclude that autonomous LLM grading is not defensible at current capability, but that
per-rubric bias calibration plus human review of flagged items is. We argue the field
should report inter-human reliability as a precondition for any claim about LLM grading
accuracy.

**Keywords.** automated grading, large language models, inter-rater reliability,
assessment, engineering education, LLM evaluation

---

## 1. Introduction

Grading open-response engineering work is expensive, slow, and — as every instructor who
has moderated a teaching team knows — inconsistent. The arrival of capable large language
models (LLMs) has therefore prompted a wave of proposals to automate it.

The published results do not agree. Kortemeyer reports R² = 0.84 against human graders on
introductory physics derivations [4]; Bernik et al. report Pearson 0.91 on programming
assignments [6]; Gobrecht et al. report an automated grader whose median absolute error is
44% *smaller* than that of certified human re-graders [3]; Tang et al. report human-AI
agreement comparable to human inter-rater reliability on physics exams [9]. Against these,
Mathew et al. report all tested models below QWK 0.30 where human raters reach 0.72 [10],
Caraeni et al. judge GPT-4o's accuracy on handwritten mathematics "too low for real-world
settings" [11], and Lundgren finds low inter-rater reliability between GPT-4 and instructors
even when average grades match [12]. These are not differences of degree; they are opposite
conclusions from competent studies.

Most of this evidence shares a design choice that we believe explains much of the
disagreement. A model's output is compared against one reference — a single instructor, a
single teaching assistant, or an answer key — and agreement with that reference is reported
as accuracy. The design is convenient, and for objective item types it is sound. For
rubric-scored open response it embeds an assumption that is rarely tested: that the reference
grader *is* the correct grade, and by extension that a second human grader would have
produced substantially the same number.

Flodén, comparing ChatGPT against the teachers who set three Master's-level exams, states the
problem precisely without being able to resolve it: having found that the model's grades
differed from the teachers' in most cases, he observes that "it is not unlikely that two
different human graders could result in similar discrepancies" [5]. Resolving that requires
several independent human graders on the same items — which most studies, including that one,
do not have.

That assumption is measurable, and when we measure it we find it does not hold uniformly.
Across our two courses, the same three-grader protocol produced inter-rater ICC(2,1) of
0.956 in one course and 0.504 in the other. In the first, human graders are nearly
interchangeable and a demanding bar for an LLM is appropriate. In the second, the humans
themselves disagree substantially, and a model judged against any one of them would be
scored largely on which human it happened to be compared against. A single number —
"correlation with the instructor" — cannot distinguish these two situations, yet they
call for opposite deployment decisions.

This paper makes four contributions.

**A method.** We describe a *human-ceiling protocol* for evaluating LLM graders
(Section 3). It requires measuring inter-human reliability before any model is scored,
constructing a consensus reference rather than privileging one grader, and then applying
an insertion test: does adding the model to the human panel as an additional rater
preserve the panel's reliability? The protocol converts the vague question "is the model
accurate?" into the answerable question "is the model within the variability the
department already tolerates among its own graders?"

**Evidence.** We apply the protocol to seven frontier models across two dissimilar
engineering assignments, with three independent human graders per assignment and three
repeated runs per model (Sections 4–6). The repeated runs let us separate a model's
systematic error from its run-to-run instability, a property invisible to single-run
studies but decisive for deployment.

**Guidance.** We show that the dominant error component is a systematic harshness offset
rather than noise, and we derive from this a set of deployment recommendations that are
neither "replace the TAs" nor "do not use LLMs" (Section 8).

**A reconciliation.** Because our two courses have very different human ceilings while
sharing one protocol and one set of models, we can reproduce the field's contradictory
verdicts within a single study (Section 8.5). We argue that much of the disagreement in the
literature is a disagreement about reference quality rather than about model capability, and
we state the prediction that would test this.

We also report, as a cautionary result, what the conventional analysis would have
concluded on the same data (Section 5.7). The difference is not marginal: five of seven
models change the sign of their reported bias, and the ranking of which models are
"significantly biased" substantially reorders. We present this not to criticise prior
work but to argue that the human ceiling is a precondition for interpreting any of these
numbers.

---

## 2. Related Work

### 2.1 Automated grading before LLMs

Automated assessment has a long history in engineering and computing education. Burrows,
Gurevych, and Stein's survey of automatic short answer grading (ASAG) identifies 35 systems
across five temporal eras, tracing the field from concept-mapping and information-extraction
approaches through corpus-based and machine-learning methods [1]. Their component analysis
established what has remained the field's most durable finding: performance is strongly
item-dependent, and the era of *evaluation* — rather than of new architectures — is what
consolidates the field. Our results in Section 5.6, where per-question MAE varies by a factor
of five across items within a single assignment, are consistent with this.

The pre-LLM ceiling was set by fine-tuned transformers. Dada et al. released a structured
dataset of open-ended university responses and found Longformer best among traditional and
transformer approaches, at Spearman 0.77 against human graders [2]. Most notably for our
purposes, Gobrecht et al. trained an ASAG model on university exam data and evaluated it
against *certified human domain experts re-grading historic exams*, reporting that the
model's median absolute error was 44% smaller than the human re-graders' — concluding that
automated grading is more consistent than humans and can therefore increase fairness [3].
This is the strongest pro-automation claim in the literature we review, and it is explicitly
ceiling-referenced. We return to it in Section 8.5.

### 2.2 LLMs as graders in higher education

Instruction-tuned LLMs removed the need for per-item training data and prompted a rapid wave
of evaluations in higher education. **The reported results are contradictory, and that
contradiction is this paper's point of departure.**

**Favourable findings.** Kortemeyer graded handwritten introductory-physics derivations with
MathPix and GPT-4, achieving R² = 0.84 against human graders, but concluded that the workflow
suits formative feedback and that for final evaluations it "would best be used to assist
human graders" [4]. Flodén conducted the study closest in design to ours: three Master's-level
exams, 463 responses, each graded at least three times by ChatGPT for 1,389 total gradings,
compared against the teachers who set them [5]. Seventy per cent of gradings fell within 10%
of the teachers' marks and 31% within 5%. Bernik et al. compared GPT-4-turbo, Claude 3, and
Gemini 1.5 Pro on 315 Python assignments, with GPT-4 achieving Pearson 0.91 and mean absolute
deviation 0.68 points [6]. Poličar et al. concluded that, with well-designed prompts, LLM
grading of a bioinformatics course was comparable to human teaching assistants in both
scoring and feedback [7]. Henkel et al. reported GPT-4 at Cohen's kappa 0.70 against human
raters at 0.75 on K-12 short-answer marking [8]. Tang, Ambrose, and Cheng scored
constructed-response physics exams using four instructors across two rounds and found
human-AI agreement on total scores *comparable to human inter-rater reliability*, though the
model struggled specifically with mid-range responses involving partial or ambiguous
reasoning [9].

**Unfavourable findings.** Mathew et al. evaluated six models on the ASAP and DREsS essay
corpora and found all below QWK 0.30 against a human-human ceiling of QWK 0.72, reporting
that LLMs compress scoring ranges — inflating underdeveloped essays while penalising minor
language errors in otherwise strong work [10]. Caraeni, Scarlatos, and Lan evaluated GPT-4o
on handwritten university mathematics exams and found that although rubrics improved
alignment, overall accuracy remained "too low for real-world settings" [11]. Lundgren found
that GPT-4 produced comparable *average* grades to human instructors while its inter-rater
reliability with those instructors was low, and that it graded risk-aversely, attending to
surface features rather than disciplinary standards [12].

Broader reviews of AI-assisted grading document both the efficiency case and the persistent
need for human oversight [13], and instructor acceptance remains an independent constraint on
adoption [14].

**Documented failure modes** include systematic leniency or severity [10], [12], compression
toward the centre of the scale [5], [10], sensitivity to prompt formulation — where
semantically equivalent prompts shift behaviour substantially [15] — and instability across
repeated invocations of an identical prompt. Stureborg et al. characterise LLM evaluators as
both inconsistent and biased, reporting skewed rating distributions, anchoring effects, and
low agreement with themselves on identical samples [16]. This last property motivates our
repeated-run design (Section 5.6); it is invisible to single-run studies.

### 2.3 LLM-as-judge and its evaluation designs

A parallel NLP literature evaluates LLMs as judges of model output rather than of student
work. Zheng et al. introduced MT-Bench and Chatbot Arena and documented position, verbosity,
and self-enhancement biases in LLM judges [17]. Notably, they benchmark GPT-4's agreement with
humans (approximately 85%) against *human-human* agreement (approximately 81%) — precisely the
ceiling-referenced comparison we argue educational studies should adopt as standard. The
insertion test of Section 3 is adapted from this tradition of treating the judge as one rater
among several rather than as an oracle.

### 2.4 Inter-rater reliability in educational assessment

Measurement theory has long held that a single rater is an unreliable instrument. Shrout and
Fleiss formalised the intraclass correlation coefficient and its six forms, distinguishing the
reliability of a single rater from that of a k-rater average [18]; Cohen's weighted kappa
extends chance-corrected agreement to ordinal scales with partial credit [19]; Hayes and
Krippendorff argue for alpha as a standard reliability measure, notably because it tolerates
missing judgements [20] — which real grading data has, as Section 5.1 demonstrates.
Generalizability theory, developed by Cronbach et al. [21] and applied to performance
assessment by Brennan [22], decomposes measurement error into facets including raters, items,
and occasions, and is the natural framework for the question we ask.

### 2.5 The gap

**Ceiling-referenced comparison is not new**, and we do not claim to have introduced it.
Gobrecht et al. [3], Henkel et al. [8], Tang et al. [9], Mathew et al. [10], and Zheng et al.
[17] all report human agreement alongside model agreement. Our contribution is narrower.

Consider what the literature above collectively asserts. Gobrecht et al. report a model
beating human re-graders by 44% [3]; Tang et al. report parity with human inter-rater
reliability [9]; Mathew et al. report QWK below 0.30 against a human 0.72 [10]. These are not
differences of degree. **They are opposite answers to the same question, produced by
competent studies using defensible statistics.**

Flodén states the resolving hypothesis explicitly but does not test it. Observing that
ChatGPT's grades differed from the teachers' in most cases, he notes that "it is not unlikely
that two different human graders could result in similar discrepancies" [5]. That is exactly
the right question — and answering it requires multiple independent human graders on the same
items, which that study, like most, did not have. The hypothesis has been available for years
without being measured.

Our contribution is therefore threefold.

1. **The insertion test.** Reporting a human ceiling *beside* a model score is weaker than
   asking whether the model, added to the existing panel, preserves that panel's reliability.
   We are not aware of this test being applied to LLM grading of engineering coursework, and
   it is the question a department actually faces when it considers deployment.
2. **A demonstration that the design choice changes the answer, on one dataset.** Section 5.7
   shows that the conventional single-reference, run-pooled analysis reverses the sign of
   measured bias for five of seven models on our data, and reverses significance verdicts in
   both directions. This converts "you should measure the ceiling" from methodological advice
   into an empirical result.
3. **Two courses, one protocol, ceilings differing by a factor of nearly two.** This lets us
   address the contradiction above directly (Section 8.5) rather than adding a further data
   point to it.

---

## 3. The Human-Ceiling Protocol

The protocol has six steps. Steps 1–2 concern the humans and must be completed before any
model is scored; steps 3–6 concern the models.

### Step 1 — Quantify the human ceiling

With three or more independent graders scoring the same items, compute at the
*question-instance* level:

- **ICC(2,1)** — reliability of a single randomly chosen rater; the quantity that matters
  when one grader will grade one submission.
- **ICC(2,k)** — reliability of the k-rater average; the quality of the consensus itself.
- **Krippendorff's alpha** (interval) — tolerates missing judgements, which real grading
  data has.
- **Kendall's W** — concordance of the rank ordering the graders impose on students.
- **Mean pairwise human-human MAE**, in points per question — the interpretable form.
  This is the number instructors actually reason about.

These jointly define the ceiling. The last is the most useful for the deployment question,
because it answers "how many points apart are two of our own graders, on average?"

### Step 2 — Form a consensus reference

Rather than privileging one grader, take the per-question **mean** across graders as the
reference, with the median computed as a robustness check. Where the two diverge, the item
is contested and should be flagged rather than averaged away. Missing judgements are
skipped in the mean, not imputed — but the resulting asymmetry must be tracked, because it
propagates (Section 5.1).

### Step 3 — Score each model against the consensus

Compute MAE, RMSE, Pearson r, Spearman rho, agreement-R², bias (signed mean error), and
error SD. Compute these **per run and then average across runs**, rather than pooling runs,
so that repeated measurements of the same item are not treated as independent observations.

### Step 4 — The insertion test

Insert the model into the panel as an additional rater and recompute ICC(2,1) and
Krippendorff's alpha over the enlarged panel. Compare to the human-only baseline. If
reliability is preserved, the model is contributing rater-quality judgements; if it drops,
the model is injecting disagreement. This is the protocol's central test, because it asks
the deployment question directly: *would we accept this as one more grader on the team?*

### Step 5 — The within-human-variability test

Compute each model's mean absolute disagreement with each individual human, and compare it
to the mean human-human disagreement. A model whose disagreement with the humans is no
larger than the humans' disagreement with each other is, by the department's own operative
standard, an acceptable grader. This yields the interpretable **ceiling ratio**:

> ceiling ratio = (model's mean MAE vs. individual humans) / (mean human-human MAE)

A ratio of 1.0 means the model is exactly as far from the humans as they are from each
other. Anything above 1.0 is a model that disagrees with the staff more than the staff
disagree among themselves.

### Step 6 — Run-to-run self-consistency

Human graders grade once; a model can be invoked repeatedly. For every
(student, question, model) triple, compute the standard deviation of scores across runs and
average. This measures grading *noise* — a distinct failure mode from bias, invisible to
run-pooled analysis, and directly relevant to fairness: a noisy grader assigns different
grades to identical work.

**Why the order matters.** Steps 1 and 2 must precede steps 3–6. Once a single grader has
been designated "the reference", every subsequent number inherits that grader's idiosyncrasy,
and there is no way to recover the ceiling after the fact.

---

## 4. Study Design

### 4.1 Grading pipeline

All models were driven through a single grading service that presents each model with an
identical task structure: for each question, the question text, an optional reference
answer, the rubric, the maximum points, and the student's answer; and requests a per-question
score with strengths, areas for improvement, and a breakdown. Questions are flattened —
multi-part questions are expanded into their sub-parts — and each submission is graded in a
single bulk call rather than one call per question, so the model sees the whole submission
in context.

Objectively-scorable item types (multiple choice, true/false) are graded deterministically
by the service rather than by the model, and are excluded from all analyses reported here.
Every number in this paper therefore concerns open-response, rubric-scored items only.

Three provider APIs were used: the OpenAI Chat Completions API, Amazon Bedrock's Converse
API for the Anthropic models, and the Google GenAI API for the Gemini models.

### 4.2 Models

| Model | Provider / API | Decoding setting |
|---|---|---|
| GPT-4o | OpenAI | temperature 0.1, max 16384 tokens |
| GPT-5 | OpenAI | reasoning effort "high" (temperature not applicable) |
| Claude Haiku 4.5 | Anthropic via Bedrock | temperature 0.1, max 2000 tokens |
| Claude Sonnet 4.6 | Anthropic via Bedrock | temperature 0.1, max 2000 tokens |
| Claude Opus 4.6 | Anthropic via Bedrock | temperature 0.1, max 2000 tokens |
| Gemini 2.5 Flash | Google GenAI | temperature 0.1, max 2000 tokens |
| Gemini 2.5 Pro | Google GenAI | temperature 0.1, max 2000 tokens |

Each model graded every submission **three times**. These decoding settings are not fully
uniform across providers, and we treat that as a limitation rather than a design feature;
see Section 9.

### 4.3 Course 1 — Operating Systems (primary study)

An undergraduate Operating Systems assignment: 6 open-response questions, 133 points total,
with per-question maxima ranging from 15 to 40 points. Forty student submissions were
graded independently by three teaching assistants (TA1, TA2, TA3), and by each of the seven
models three times. Student answers were supplied to the models as structured text.

This yields 240 question instances, 680 human question-level judgements (720 less the 40
missing TA2 Q6 scores, Section 5.1), and 5,040 model question-level scores
(240 x 7 models x 3 runs).

### 4.4 Course 2 — Biomaterials (contrast study)

A Biomaterials assignment: 5 open-response questions, 2 points each, 10 points total. Five
student submissions were graded independently by three human graders, and by each of the
seven models three times. Submissions were supplied to the models as **PDFs ingested
directly**, rather than as extracted structured text.

This yields 25 question instances, 75 human judgements, and 525 model scores.

**On the role of this study.** With five students, this arm is underpowered and we do not
treat it as independent confirmation of the Operating Systems results; its confidence
intervals are wide and overlapping (Section 6). It is included because its human ceiling is
*low* — ICC(2,1) = 0.504 against 0.956 for Operating Systems — which makes it the more
informative case for the protocol itself. It demonstrates that the bar an LLM must clear is
a property of the course and its graders, not a universal constant.

### 4.5 Analysis

Both courses were analysed with the identical protocol of Section 3. Confidence intervals
are bootstrap (percentile) intervals. Paired comparisons between a model and the human
consensus use the run-averaged score per student as the unit, giving n = 40 and n = 5
respectively, and are reported with both paired t-tests and Wilcoxon signed-rank tests
alongside Cohen's d. Rank correlations across the two courses are Spearman's rho over the
seven models.

---

## 5. Study 1 — Operating Systems

### 5.1 A data-quality finding that changes the results

Before reporting agreement, one property of the human data must be stated, because it
propagates into every naive comparison.

**TA2 did not grade Question 6.** All forty of TA2's Q6 scores are missing. TA2's recorded
total is therefore a sum over Q1–Q5 (out of 93) while TA1's and TA3's are sums over Q1–Q6
(out of 133). Any analysis that averages the three recorded totals to form a reference
produces a reference that is approximately 8.8 points too low — because one third of the
average is missing an entire question worth 40 points.

The consequence is that every model appears roughly 8.8 points *more generous* than it is.
As Section 5.7 shows, this is enough to reverse the sign of the reported bias for five of
the seven models. We handle it by computing the consensus per question instance (so the Q6
consensus is the TA1/TA3 mean), by computing ICC and Kendall's W on the fully-crossed
Q1–Q5 subset, and by using Krippendorff's alpha — which tolerates missingness — on all 240
instances.

We report this in detail because it is not an exotic failure. Partial grading, split
grading duties, and missing rubric rows are ordinary features of real course data, and a
single-reference pipeline absorbs them silently.

### 5.2 The human ceiling

**Table 1 — Inter-human agreement, Operating Systems.**

| Measure | Value |
|---|---|
| ICC(2,1), single rater (200 fully-crossed units) | **0.956** |
| ICC(2,k), 3-TA average | 0.985 |
| Krippendorff's alpha, interval (240 units) | 0.956 |
| Kendall's W | 0.969 |
| Mean pairwise TA-TA MAE | **1.113 pts/question** |
| — TA1–TA2 / TA1–TA3 / TA2–TA3 | 1.10 / 1.53 / 0.70 |
| Mean pairwise quadratic-weighted kappa (grade bins, Q1–Q5) | 0.896 |
| Mean pairwise TA-TA MAE, total score (Q1–Q5, /93) | 3.46 pts |
| TA mean totals (Q1–Q5, /93) | TA1 56.5, TA2 55.6, TA3 53.9 |

These TAs are close to interchangeable. ICC(2,1) = 0.956 means a single randomly chosen TA
already reproduces the panel almost exactly, and the three TAs' mean totals span 2.6 points
out of 93. **The bar for a model in this course is therefore high, and it is high for a
principled reason, not because of a convention that ICC should exceed 0.9.**

### 5.3 Models against the consensus

The mean and median consensus agree closely (r = 0.998); we use the mean.

**Table 2 — Question-level metrics vs. the TA consensus (240 instances, per-run averaged).
Reference: TA-TA MAE = 1.113 pts/question.**

| Model | MAE | RMSE | Pearson r | Spearman rho | R² | Bias | Error SD |
|---|---|---|---|---|---|---|---|
| Claude Opus | **3.148** | **4.691** | **0.879** | **0.818** | **0.753** | −0.719 | **4.645** |
| Claude Sonnet | 3.316 | 4.853 | 0.872 | 0.811 | 0.736 | −1.077 | 4.742 |
| Gemini 2.5 Pro | 3.594 | 5.739 | 0.831 | 0.747 | 0.628 | **−0.343** | 5.739 |
| Gemini 2.5 Flash | 3.725 | 5.813 | 0.829 | 0.750 | 0.621 | −0.411 | 5.807 |
| Claude Haiku | 3.945 | 6.192 | 0.804 | 0.762 | 0.570 | −2.253 | 5.779 |
| GPT-5 | 3.956 | 6.145 | 0.810 | 0.721 | 0.577 | −0.904 | 6.090 |
| GPT-4o | 4.891 | 6.868 | 0.766 | 0.652 | 0.472 | −1.595 | 6.691 |

Read on its own, this table looks like a success. Correlations of 0.77–0.88 are in the
range routinely reported as evidence that LLMs can grade. **The MAE column tells a different
story.** The best model is 3.148 points per question away from the consensus, against a
human-human disagreement of 1.113 — nearly three times as far. Correlation is high because
the models rank students correctly; the absolute grades are not close.

Every bias is negative. All seven models grade this assignment harshly.

*[Figure 1: `Grading_Dataset_OS/outputs/4_2_llm_vs_ta_scatter.png` — per-model scatter of
model score against TA consensus.]*

*[Figure 2: `Grading_Dataset_OS/outputs/5_4_bland_altman.png` — Bland–Altman agreement.
The mean-difference lines sit below zero across the score range, confirming that the
disagreement is a systematic offset and not a scale-dependent distortion.]*

### 5.4 The insertion test

**Table 3 — ICC(2,1) with each model inserted as a fourth rater. Human-only baseline:
ICC(2,1) = 0.956, Krippendorff's alpha = 0.956.**

| Model | ICC(2,1) with model | ICC(2,k) | Krippendorff's alpha | Change vs. human-only |
|---|---|---|---|---|
| Claude Opus | 0.865 | 0.962 | 0.903 | **−0.091** |
| Claude Sonnet | 0.864 | 0.962 | 0.897 | −0.092 |
| Gemini 2.5 Pro | 0.826 | 0.950 | 0.873 | −0.130 |
| Claude Haiku | 0.824 | 0.949 | 0.847 | −0.132 |
| Gemini 2.5 Flash | 0.815 | 0.946 | 0.872 | −0.141 |
| GPT-5 | 0.805 | 0.943 | 0.859 | −0.151 |
| GPT-4o | 0.747 | 0.922 | 0.829 | −0.209 |

No model preserves panel reliability. Adding the best available model to this teaching team
costs 0.091 of ICC; adding the weakest costs 0.209. In the language a department would use:
every one of these models is a *worse-than-average* member of this grading team, and
replacing a TA with any of them measurably degrades the consistency of grading students
receive.

*[Figure 3: `docs/paper/figures/fig3_icc_as_fourth_rater_os.png`]*

### 5.5 The within-human-variability test

**Table 4 — Model disagreement with each individual TA, points per question.
TA-TA ceiling = 1.113.**

| Model | vs. TA1 | vs. TA2 | vs. TA3 | Mean | Ceiling ratio | Within human variability? |
|---|---|---|---|---|---|---|
| Claude Opus | 3.485 | 2.787 | 3.074 | **3.115** | **2.80x** | No |
| Claude Sonnet | 3.707 | 2.887 | 3.169 | 3.254 | 2.92x | No |
| Gemini 2.5 Pro | 3.874 | 3.103 | 3.461 | 3.480 | 3.13x | No |
| Gemini 2.5 Flash | 3.947 | 3.347 | 3.640 | 3.644 | 3.27x | No |
| Claude Haiku | 4.252 | 3.247 | 3.819 | 3.773 | 3.39x | No |
| GPT-5 | 4.149 | 3.581 | 3.883 | 3.871 | 3.48x | No |
| GPT-4o | 5.060 | 4.645 | 4.770 | 4.825 | 4.34x | No |

Zero of seven models fall within human variability. The gap is not marginal — the best
model would need to reduce its disagreement by 64% to reach the ceiling.

*[Figure 4: `Grading_Dataset_OS/outputs/9_5_within_human_variability.png`]*

### 5.6 Where the models fail, and how noisily

Per-question analysis locates the error. Q6 — the 40-point question, and the one TA2 did
not grade — is the hardest for every model: Claude Haiku's MAE on Q6 is 7.76 points against
1.47 on Q4. Q4, the most constrained item, is where every model does best (MAE 1.31–1.58).
The pattern is consistent with the pre-LLM autograding literature: constrained items
automate well, open-ended synthesis does not.

*[Figure 5: `Grading_Dataset_OS/outputs/8_2_per_question_mae_heatmap.png`]*

**Table 5 — Run-to-run standard deviation (points per question, mean over all
student-question pairs).**

| Model | Run-to-run SD |
|---|---|
| Claude Opus | **0.765** |
| Claude Sonnet | 0.883 |
| Claude Haiku | 0.994 |
| GPT-5 | 1.027 |
| GPT-4o | 1.147 |
| Gemini 2.5 Pro | 1.376 |
| Gemini 2.5 Flash | 1.582 |

This is a fairness result, not just a reliability one. Gemini 2.5 Flash, invoked twice on
identical work, produces scores differing by 1.58 points per question on average — 9.5
points across a six-question assignment. The models with the lowest noise are the two that
also lead on accuracy, but the ordering is not identical: GPT-4o is mid-pack on noise
despite being last on accuracy, and Gemini 2.5 Pro is third on accuracy but second-noisiest.

*[Figure 6: `Grading_Dataset_OS/outputs/9_6_run_to_run_noise.png`]*

### 5.7 What the conventional analysis would have concluded

We now run the analysis the conventional way on the same data: reference = the mean of the
three recorded TA totals (inheriting the TA2/Q6 gap), and each (student, run) row treated as
an independent observation (n = 120 rather than n = 40).

**Table 6 — Naive vs. protocol-corrected bias, Operating Systems. Negative bias = harsher
than the human consensus.**

| Model | Naive bias | Corrected bias | Sign flip? | Naive p | Corrected p (n=40) | Cohen's d | Corrected verdict |
|---|---|---|---|---|---|---|---|
| Claude Haiku | −4.70 | −13.52 | — | 0.001 | <0.0001 | −0.908 | Significantly harsh |
| GPT-4o | −0.75 | −9.57 | — | **0.667** | **0.0013** | −0.548 | Significantly harsh |
| Claude Sonnet | **+2.35** | −6.46 | **Yes** | 0.047 | 0.0038 | −0.530 | Significantly harsh |
| GPT-5 | **+3.39** | −5.42 | **Yes** | 0.023 | 0.0366 | −0.359 | Significantly harsh |
| Claude Opus | **+4.50** | −4.31 | **Yes** | <0.001 | 0.0467 | −0.366 | Significantly harsh |
| Gemini 2.5 Flash | **+6.35** | −2.47 | **Yes** | **<0.001** | **0.2162** | −0.195 | Not significant |
| Gemini 2.5 Pro | **+6.76** | −2.06 | **Yes** | **<0.001** | **0.3011** | −0.166 | Not significant |

The conventional analysis of this dataset would have supported the following claims, all of
which are artefacts:

1. **That five of seven models grade *generously*.** They do not; all seven grade harshly.
   The sign reversal is driven entirely by the missing-TA2-Q6 reference contamination.
2. **That Gemini 2.5 Pro and Flash exhibit highly significant bias (p < 0.001).** Corrected,
   these are the two models whose bias is *not* statistically distinguishable from zero
   (p = 0.30 and p = 0.22) — they are the best-calibrated models in the study.
3. **That GPT-4o is unbiased (p = 0.667).** Corrected, GPT-4o is significantly harsh
   (p = 0.0013, d = −0.548).

Two distinct errors produce this. Reference contamination shifts every bias estimate by a
constant. Pseudoreplication — treating three runs of the same student as three independent
observations — inflates n threefold and shrinks p-values, manufacturing significance for
effects that are small relative to between-student variance.

Neither error is exotic, and neither is visible from within the naive analysis. Only the
step-1 requirement to characterise the human panel *before* scoring any model surfaces the
first, and only run-averaging surfaces the second.

---

## 6. Study 2 — Biomaterials

### 6.1 A different ceiling

**Table 7 — Inter-human agreement, Biomaterials, with Operating Systems for comparison.**

| Measure | Biomaterials | Operating Systems |
|---|---|---|
| ICC(2,1), single rater | **0.504** [0.220, 0.730] | 0.956 |
| ICC(2,k) | 0.753 | 0.985 |
| Krippendorff's alpha | 0.469 | 0.956 |
| Kendall's W | 0.505 | 0.969 |
| Mean pairwise weighted kappa | 0.516 | 0.896 |
| Mean human-human MAE | **0.227 pts/q** [0.133, 0.320] | 1.113 pts/q |
| Human mean totals (/10) | H1 9.30, H2 7.80, H3 9.10 | — |

The same three-grader protocol, applied to a different course, yields a fundamentally
different instrument. ICC(2,1) = 0.504 is moderate at best, and the graders' mean totals
span 1.5 points out of 10 — H2 is a markedly harsher grader than H1 and H3.

Two factors contribute. The rubric is coarse: five questions at 2 points each, with most
human scores falling in [1.5, 2.0]. This range restriction mechanically depresses ICC, which
is a ratio of between-subject to total variance — when students genuinely differ little,
even small rater disagreements dominate. It also drives the negative R² values in Table 8:
with almost no variance around the consensus to explain, any deviation makes a model worse
than predicting the mean. Under range restriction, R² is the wrong summary and MAE is the
right one.

**This is the point of including the study.** Had we evaluated a model here against H2, it
would look accurate; against H1, harsh. The ceiling framework makes the ambiguity explicit
instead of hiding it in the choice of reference grader.

### 6.2 Models against the consensus

**Table 8 — Question-level metrics vs. consensus (25 instances, 2 points max per question).
Reference: human-human MAE = 0.227.**

| Model | MAE | MAE 95% CI | RMSE | Pearson r | Bias | Bias 95% CI | Cohen's d |
|---|---|---|---|---|---|---|---|
| Claude Haiku | **0.339** | [0.25, 0.44] | **0.416** | 0.464 | **−0.265** | [−0.39, −0.14] | −0.809 |
| Claude Opus | 0.385 | [0.28, 0.49] | 0.465 | 0.671 | −0.372 | [−0.48, −0.26] | −1.310 |
| Claude Sonnet | 0.386 | [0.29, 0.49] | 0.461 | 0.635 | −0.373 | [−0.49, −0.27] | −1.348 |
| Gemini 2.5 Flash | 0.419 | [0.29, 0.58] | 0.554 | **0.690** | −0.405 | [−0.56, −0.27] | −1.054 |
| Gemini 2.5 Pro | 0.423 | [0.27, 0.60] | 0.594 | 0.634 | −0.397 | [−0.58, −0.23] | −0.880 |
| GPT-4o | 0.507 | [0.41, 0.60] | 0.558 | 0.616 | −0.493 | [−0.60, −0.39] | −1.857 |
| GPT-5 | 0.544 | [0.40, 0.70] | 0.663 | 0.590 | −0.524 | [−0.68, −0.37] | −1.263 |

**Every confidence interval on bias excludes zero.** All seven models are significantly
harsh (paired t and Wilcoxon both p ≤ 0.001), with large effect sizes (|d| = 0.81–1.86).
On a 10-point assignment, the human graders average 8.7 and the models average 6.1–7.4 —
a gap of well over a letter grade.

The MAE intervals overlap heavily. With 25 question instances, the ordering of adjacent
models in this study is **not** statistically resolved, and we do not claim it is. What is
resolved is the direction: every model, harsh, with a large effect size.

*[Figure 7: `Biomaterials/outputs/04_bias_by_model.png`]*

*[Figure 8: `Biomaterials/outputs/10g_radar.png`]*

### 6.3 Insertion, variability, and noise

The human-only baseline is ICC(2,1) = 0.504. With a model inserted as a fourth grader:
Gemini 2.5 Flash 0.426, Claude Opus 0.425, Claude Sonnet 0.408, Gemini 2.5 Pro 0.407,
Claude Haiku 0.397, GPT-5 0.347, GPT-4o 0.342 — drops of 0.077 to 0.161.

The verdict matches Operating Systems: no model preserves panel reliability, even against a
panel that is itself only moderately reliable.

*[Figure 9: `docs/paper/figures/fig9_icc_as_fourth_rater_bio.png`]*

On the within-human test, model MAE against individual humans ranges from 0.399 (Claude
Haiku) to 0.554 (GPT-5), against a ceiling of 0.227 — ceiling ratios of 1.76x to 2.44x.
Again, zero of seven within human variability.

Run-to-run SD ranges from 0.066 (Claude Opus) to 0.208 (GPT-4o) points per question, on a
2-point scale — proportionally 3.3%–10.4% of the item's value. The ordering of models by
noise is similar to Operating Systems: the Claude models are the most stable, GPT-4o and
Gemini 2.5 Flash the least.

### 6.4 One instructive item

Per-question analysis flags Q4 as an *interesting failure*: the humans agreed on it more
than on any other item (SD 0.115) and the consensus was 1.9/2.0, yet the models' mean
absolute error was 0.444 — their second-worst. An item that is unambiguous to every human
grader is one the models systematically misread. Conversely on Q5 the humans were most
split (SD 0.273, consensus 1.5) while the models were confidently wrong in a consistent
direction.

Both patterns argue against using model-human disagreement as an automatic flag for a
contested item: the two do not track each other.

---

## 7. Cross-Domain Synthesis

### 7.1 What replicates, stated precisely

The two courses differ in discipline, rubric granularity (2-point vs. 40-point items),
scale (10 vs. 133 points), cohort size, input modality (PDF vs. structured text), and human
ceiling (0.504 vs. 0.956). Against that, we ask what survives.

**Rank correlation across the two courses (Spearman's rho, n = 7 models):**

| Metric | rho | p |
|---|---|---|
| Composite score | 0.750 | 0.052 |
| ICC as 4th rater | 0.643 | 0.119 |
| Run-to-run SD | 0.679 | 0.094 |
| MAE | 0.571 | 0.180 |

**The full ranking does not replicate.** None of these correlations reaches significance at
n = 7, and we explicitly decline the claim that our model ordering generalises. Reporting it
as a stable leaderboard would not be supported by this data.

What does replicate:

1. **The top two, exactly.** Claude Opus is rank 1 and Claude Sonnet rank 2 in both courses,
   on every individual metric we computed.
2. **The OpenAI models are bottom-tier in both**, though they trade places — GPT-4o last on
   Operating Systems, GPT-5 last on Biomaterials.
3. **The middle does not order consistently.** Gemini 2.5 Pro is 3rd on Operating Systems
   and 5th on Biomaterials; Claude Haiku is 6th and 4th respectively.
4. **Both verdicts replicate without qualification.** Zero of seven models within human
   variability, in both courses. Every model harsh, in both courses — twelve of fourteen
   model-course pairs significantly so, the two exceptions being Gemini 2.5 Pro and Flash
   on Operating Systems.

### 7.2 The ceiling ratio

**Table 9 — Ceiling ratios: model disagreement with humans, in units of human-human
disagreement.**

| Model | Operating Systems | Biomaterials |
|---|---|---|
| Claude Opus | 2.80x | 1.80x |
| Claude Sonnet | 2.92x | 1.85x |
| Gemini 2.5 Pro | 3.13x | 1.98x |
| Gemini 2.5 Flash | 3.27x | 1.94x |
| Claude Haiku | 3.39x | **1.76x** |
| GPT-5 | 3.48x | 2.44x |
| GPT-4o | 4.34x | 2.31x |

*[Figure 10: `docs/paper/figures/fig10_cross_domain_ceiling_ratio.png`]*

Every bar in both courses is above 1.0. The ratios are *lower* in Biomaterials not because
the models grade it better in absolute terms — they do not — but because the humans grade
it less consistently, lowering the bar. This is the framework's central point rendered
numerically: the same model can be 2.80x or 1.80x from acceptable depending entirely on
whose grading it is being asked to match.

It also implies a practical corollary. **The courses where an LLM grader is most likely to
clear the human bar are exactly the courses where human grading is least reliable** — which
is a much weaker endorsement than "the LLM is accurate", and should be described honestly
as such when deploying.

### 7.3 Bias dominates variance

Across both courses, the signed bias accounts for most of the mean absolute error. On
Biomaterials, mean bias is −0.40 points/question against a mean MAE of 0.43 — the models are
almost *entirely* offset, with comparatively little residual scatter. On Operating Systems
the offset is a smaller share of the total, but is still statistically significant for five
of seven models.

This is the most actionable finding in the paper, and Section 8.1 develops it.

### 7.4 Latency

**Table 10 — Grading latency per submission (seconds).**

| Model | OS (median) | Biomaterials (mean) |
|---|---|---|
| GPT-4o | **8.2** | **7.4** |
| Gemini 2.5 Flash | 25.9 | 28.5 |
| Claude Haiku | 27.0 | 20.0 |
| Gemini 2.5 Pro | 37.6 | 31.3 |
| Claude Opus | 52.6 | 37.9 |
| Claude Sonnet | 54.2 | 38.8 |
| GPT-5 | 134.2 | 88.0 |

Latency spans a factor of 16 and is *inversely* related to accuracy at both extremes: the
fastest model is the least accurate, and the slowest (GPT-5) is mid-pack. Claude Opus
delivers the best accuracy at roughly 40% of GPT-5's latency. For a 40-student assignment,
even the slowest configuration completes a full grading pass in under 90 minutes
unattended, so latency is unlikely to be the binding constraint in practice.

*[Figure 11: `Grading_Dataset_OS/outputs/4_6_latency_boxplot.png`]*

---

## 8. Discussion

### 8.1 The error is an offset, and offsets are correctable

The strongest practical implication follows from Section 7.3. A grader that is wrong at
random is unfixable without improving the grader. A grader that is *consistently* wrong in
one direction is a calibration problem.

On Biomaterials, subtracting each model's mean bias from its scores would reduce mean
absolute error from roughly 0.43 to roughly the residual error SD — the models' rank
ordering of students is already reasonable, and the harshness offset is what puts the
absolute grades outside acceptable range. A per-rubric calibration constant, estimated from
a modest sample of double-graded submissions, addresses this directly.

We emphasise the constraints. The offset is *per rubric*, not per model: Claude Haiku's bias
is −2.253 points/question on Operating Systems and −0.265 on Biomaterials, and these are not
convertible into each other by rescaling. Calibration therefore requires human-graded
submissions from the same assignment, which reintroduces exactly the human effort
automation was meant to remove — though at a sample size far below full grading. And
calibration cannot repair the run-to-run noise of Section 5.6, which is a separate defect.

### 8.2 What we do not recommend

**Autonomous grading of record.** Not supported by this data, in either course. Zero of
fourteen model-course pairs fall within human variability, and every model degrades panel
reliability when inserted. The gap is large, not marginal.

**Substituting a model for a TA on a grading team.** This is precisely what the insertion
test evaluates, and every model fails it.

**Trusting a high correlation.** Table 2 is the cautionary case: r = 0.879 alongside an MAE
2.80x the human ceiling. Correlation measures whether the model ranks students correctly;
it is nearly blind to a uniform offset. A model can correlate at 0.88 with the TAs while
awarding every student a grade one band too low. Any evaluation reporting correlation
without absolute error, referenced to human-human disagreement, is under-reporting.

### 8.3 What the data does support

**Second-opinion flagging.** Use the model as an additional, non-authoritative rater and
surface submissions where model and human diverge most. The model does not need to be
correct for this to be valuable — it needs to be uncorrelated enough with the human's error
to catch slips. Section 6.4 is a caution here: model-human divergence did not track item
contentiousness, so this should be validated per course rather than assumed.

**Triage for grading order.** Model scores correlate 0.77–0.88 with the consensus, which is
ample for ordering a grading queue so that borderline submissions reach a human first.

**Draft feedback.** The models produce per-question strengths, weaknesses, and breakdowns.
Nothing in our data speaks to the quality of that prose — we evaluated only scores — but the
grading task and the feedback task fail differently, and a model too harsh by a consistent
offset may still write useful formative comments. **This needs separate evaluation.**

**Rubric quality diagnosis.** The Biomaterials ceiling of 0.504 was discovered by running
step 1 of the protocol. That finding is independently valuable to the instructor,
irrespective of any LLM: it says the rubric does not discriminate reliably between graders.
Running the human half of this protocol is worthwhile even if no model is ever deployed.

### 8.4 Model selection, if deploying

Claude Opus 4.6 and Claude Sonnet 4.6 lead on every metric in both courses and are the two
most run-stable models. Sonnet achieves near-identical accuracy to Opus at comparable
latency. GPT-4o was last or near-last on accuracy in both courses, and its speed advantage
does not compensate. Gemini 2.5 Pro and Flash were the *best-calibrated* models on
Operating Systems (the only two with statistically non-significant bias) but the noisiest
across runs — a combination that suits ensemble averaging, where repeated sampling
suppresses noise while good calibration is retained. We did not test ensembling, and flag
it as the most promising untested configuration suggested by our results.

### 8.5 Reconciling the contradictory literature

Section 2.2 set out an unresolved disagreement: Gobrecht et al. report an automated grader
beating human re-graders by 44% on median absolute error [3], Tang et al. report parity with
human inter-rater reliability [9], while Mathew et al. report QWK below 0.30 against a human
ceiling of 0.72 [10], and Caraeni et al. judge accuracy too low for deployment [11].

Our two studies reproduce that disagreement *within a single paper, using one protocol and
one set of models*. Ranked against the Operating Systems TAs, the best model sits 2.80x
outside human variability and looks clearly unfit. Ranked against the Biomaterials graders,
the same model sits 1.80x outside — still failing, but by a margin that a slightly different
rubric or a slightly noisier panel would erase. The models did not change. The graders did.

This suggests a specific reading of the literature. Studies reporting parity or better tend
to involve reference conditions with high human variability: re-grading historic exams
without the original grading context [3], or rubrics on which instructors themselves diverge.
Studies reporting failure tend to involve reference conditions with well-controlled human
agreement — essay corpora with trained, calibrated raters reaching QWK 0.72 [10]. **The
apparent disagreement about model capability may be substantially a disagreement about
reference quality.**

We advance this as an explanation consistent with our data, not as a demonstrated one: we
cannot recompute other authors' ceilings, and we do not claim their conclusions are wrong.
The testable prediction is that reported LLM-human agreement should correlate negatively with
the inter-human reliability of each study's reference panel. That prediction could be checked
by a meta-analysis, and we suggest it as future work.

The practical implication is uncomfortable but important, and it restates Section 7.2: an LLM
grader is most likely to clear the human bar precisely where human grading is least reliable.
"The model performs as well as our graders" and "our graders do not agree with each other"
are compatible statements, and only the first tends to get reported.

---

## 9. Threats to Validity

**The prompt is not identical across providers.** The grading service emits a compact
delimiter-based format for Gemini models, including an explicit instruction to keep each
field under 30 words, and a JSON format for OpenAI and Anthropic models. The Gemini results
are therefore confounded with a different prompt and a tighter output-length constraint.
Gemini's cross-model comparisons should be read with this in mind; the within-Gemini
findings (calibration, noise) are unaffected.

**Decoding settings are not uniform.** Six models ran at temperature 0.1; GPT-5 ran with
high reasoning effort and no temperature control. GPT-5's run-to-run SD is therefore not
measured under the same conditions as the others, and its noise figure is not directly
comparable.

**Token budgets differ** — 2000 max tokens for Anthropic and Gemini models against 16384 for
non-reasoning OpenAI models — which could truncate long rubric feedback asymmetrically.

**Input modality is confounded with course.** Operating Systems answers were supplied as
structured text; Biomaterials submissions as directly-ingested PDFs. Any cross-course
difference may be a modality effect rather than a course effect. This does not affect the
within-course results, which are the basis for all our primary claims.

**The Biomaterials study is underpowered.** Five students, 25 question instances, wide
overlapping confidence intervals. We treat it as a contrast case for the protocol, not as
independent replication, and we do not claim its model ordering is resolved.

**The consensus is a reference, not ground truth.** Averaging three graders does not produce
correctness. Where the humans agree at ICC 0.504, the consensus is itself an unreliable
target, and a model penalised against it may not be wrong.

**Range restriction on Biomaterials.** Five 2-point items with most scores in [1.5, 2.0]
mechanically depresses ICC and produces negative R². We report MAE alongside for this
reason, but the ICC comparison across the two courses should not be read as a pure
difference in grader skill.

**One assignment per course, one institution.** Both studies are single assignments from a
single institution with a single rubric each. Question-level findings (Section 5.6) in
particular may not transfer.

**Human graders were not blinded** to the study, and grading order was not randomised.

**Model versions are a snapshot.** These are specific model versions accessed over a bounded
period; provider-side updates can change behaviour without notice. Absolute numbers should
be treated as dated; the protocol is what we intend to be durable.

**Cost was not measured.** We report latency but not API cost per submission, which is
likely a binding constraint at scale and which varies by more than an order of magnitude
across the models tested.

---

## 10. Conclusion

We evaluated seven frontier language models as rubric graders on two engineering
assignments, each graded independently by three humans, with every model run three times —
945 model gradings in total.

Our central methodological claim is that **inter-human agreement must be measured before any
claim about LLM grading accuracy is interpretable**, and we demonstrated this rather than
asserting it. On the same Operating Systems data, a conventional single-reference,
run-pooled analysis reverses the sign of measured bias for five of seven models, reports
highly significant bias for the two best-calibrated models, and reports no significant bias
for a model that is in fact significantly harsh. The two courses' human ceilings differ by a
factor of nearly two in ICC, so there is no universal bar an LLM grader can be held to. This
also gives us a candidate explanation for why the published literature reaches opposite
verdicts on LLM grading: with the models and protocol held fixed, changing only the human
panel moves the best model from 2.80x to 1.80x outside human variability. We suggest the
field's disagreement is substantially about reference quality, and note the meta-analytic
prediction that would test it.

Our central empirical claim is that **no model tested reaches human variability in either
course.** The best model disagrees with the human panel 2.80x as much as the humans disagree
with each other on Operating Systems and 1.76x on Biomaterials, and inserting any model as
an additional rater lowers panel reliability in both. This holds despite Pearson correlations
of up to 0.88, which we take as evidence that correlation alone is an inadequate reporting
standard for grading applications.

Our central practical claim is that **the dominant failure is a systematic harshness offset
rather than random error** — twelve of fourteen model-course pairs are significantly harsh —
and that this is correctable by per-rubric calibration in a way that random error would not
be. This supports assistive deployments (triage, second-opinion flagging, calibrated
first-pass scoring under human review) and does not support autonomous grading of record.

We recommend that studies of LLM grading report, as a minimum: inter-human reliability for
the same items, absolute error expressed relative to human-human disagreement, and
run-to-run variability across repeated invocations. The first is the ceiling, the second is
the honest accuracy measure, and the third is a fairness property that single-run studies
cannot see.

---

## Reproducibility

All analyses derive from two Jupyter notebooks applying the identical protocol:
`Grading_Dataset_OS/llm_vs_human_grading_analysis.ipynb` (Phase 9 implements the
human-ceiling protocol) and `Biomaterials/multiLLM_multiHuman_analysis.ipynb`. The grading
pipeline is `grading_service.py`; the experiment drivers are
`Grading_Dataset_OS/test_grading_rerun.py` and `Biomaterials/grade_biomaterials.py`.
Figures 6 and 9 are produced by `docs/paper/make_paper_figures.py`. Grades, both human and
model, are in `Grading_Dataset_OS/consolidated_results_rerun_4/` and
`Biomaterials/grading_results/`.

## Acknowledgements

[ACKNOWLEDGEMENTS]

## References

[1] S. Burrows, I. Gurevych, and B. Stein, "The eras and trends of automatic short answer
grading," *International Journal of Artificial Intelligence in Education*, vol. 25, no. 1,
pp. 60–117, 2015.

[2] I. D. Dada, A. T. Akinwale, and T.-J. Tunde-Adeleke, "A structured dataset for automated
grading: From raw data to processed dataset," *Data*, vol. 10, no. 6, art. 87, 2025.
doi: 10.3390/data10060087

[3] A. Gobrecht, F. Tuma, M. Möller, T. Zöller, M. Zakhvatkin, A. Wuttig, H. Sommerfeldt, and
S. Schütt, "Beyond human subjectivity and error: A novel AI grading system," arXiv:2405.04323,
2024. doi: 10.48550/arXiv.2405.04323

[4] G. Kortemeyer, "Toward AI grading of student problem solutions in introductory physics: A
feasibility study," *Physical Review Physics Education Research*, vol. 19, art. 020163, 2023.
doi: 10.1103/PhysRevPhysEducRes.19.020163

[5] J. Flodén, "Grading exams using large language models: A comparison between human and AI
grading of exams in higher education using ChatGPT," *British Educational Research Journal*,
vol. 51, pp. 201–224, 2025. doi: 10.1002/berj.4069

[6] A. Bernik, D. Radošević, and A. Čep, "A comparative study of large language models in
programming education: Accuracy, efficiency, and feedback in student assignment grading,"
*Applied Sciences*, vol. 15, no. 18, art. 10055, 2025. doi: 10.3390/app151810055

[7] P. G. Poličar, M. Špendl, T. Curk, and B. Zupan, "Automated assignment grading with large
language models: Insights from a bioinformatics course," *Bioinformatics*, vol. 41,
suppl. 1, pp. i21–, 2025. doi: 10.1093/bioinformatics/btaf196

[8] O. Henkel, A. Boxer, L. Hills, and B. Roberts, "Can large language models make the grade?
An empirical study evaluating LLMs' ability to mark short answer questions in K-12 education,"
arXiv:2405.02985, 2024.

[9] X. Tang, G. A. Ambrose, and Y. Cheng, "Designing reliable LLM-assisted rubric scoring for
constructed responses: Evidence from physics exams," arXiv:2604.12227, 2026.

[10] J. G. Mathew, S. Taher, A. Kundu, and D. Barbosa, "LLMs do not grade essays like humans,"
arXiv:2603.23714, 2026.

[11] A. Caraeni, A. Scarlatos, and A. Lan, "Evaluating GPT-4 at grading handwritten solutions
in math exams," in *Proc. 15th International Learning Analytics and Knowledge Conference
(LAK25)*, 2025. arXiv:2411.05231

[12] M. Lundgren, "Large language models in student assessment: Comparing ChatGPT and human
graders," arXiv:2406.16510, 2024.

[13] J. Gnanaprakasam and R. Lourdusamy, "The role of AI in automating grading: Enhancing
feedback and efficiency," IntechOpen, 2024. doi: 10.5772/intechopen.1005025

[14] R. Bello, "Exploring the role of artificial intelligence in higher education: A
comparative study on grading methods and the technology acceptance model," in *Association of
Marketing Theory and Practice Proceedings 2025*, art. 17, 2025.

[15] [AUTHORS — VERIFY], "ProSA: Assessing and understanding the prompt sensitivity of LLMs,"
in *Findings of the Association for Computational Linguistics: EMNLP 2024*, 2024.
arXiv:2410.12405

[16] R. Stureborg, D. Alikaniotis, and Y. Suhara, "Large language models are inconsistent and
biased evaluators," arXiv:2405.01724, 2024.

[17] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li,
E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, "Judging LLM-as-a-judge with MT-Bench and
Chatbot Arena," in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023),
Datasets and Benchmarks Track*, 2023. arXiv:2306.05685

[18] P. E. Shrout and J. L. Fleiss, "Intraclass correlations: Uses in assessing rater
reliability," *Psychological Bulletin*, vol. 86, no. 2, pp. 420–428, 1979.

[19] J. Cohen, "Weighted kappa: Nominal scale agreement with provision for scaled disagreement
or partial credit," *Psychological Bulletin*, vol. 70, no. 4, pp. 213–220, 1968.

[20] A. F. Hayes and K. Krippendorff, "Answering the call for a standard reliability measure
for coding data," *Communication Methods and Measures*, vol. 1, no. 1, pp. 77–89, 2007.

[21] L. J. Cronbach, G. C. Gleser, H. Nanda, and N. Rajaratnam, *The Dependability of
Behavioral Measurements: Theory of Generalizability for Scores and Profiles*. New York: Wiley,
1972.

[22] R. L. Brennan, "Performance assessments from the perspective of generalizability theory,"
*Applied Psychological Measurement*, vol. 24, no. 4, 2000.
doi: 10.1177/01466210022031796

---

> **Verification note.** All references except [15] were confirmed against the source PDF in
> `Lit/` or the publisher/arXiv record. Reference [15] needs its author list completed —
> the title, venue, and arXiv identifier are confirmed but the authors were not verified.
> Page numbers for [7] and [22] should be completed from the publisher record.
