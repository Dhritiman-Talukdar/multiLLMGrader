# Human Agreement Is the Ceiling: Evaluating Seven Frontier LLMs as Rubric Graders Across Two Engineering Courses

**Authors.** [AUTHOR LIST]
**Affiliation.** [AFFILIATION]
**Corresponding author.** [EMAIL]

---

## Abstract

Large language models are increasingly proposed as graders for open-response engineering
coursework. The published evidence contradicts itself. Reports range from automated graders
outperforming certified human re-graders to models falling far below human inter-rater
agreement on comparable tasks. Most of that evidence comes from comparing a model against a
single reference grader, a design that quietly assumes human graders agree with one another.
We show that the assumption largely settles the conclusion before any model is scored.

We evaluate seven frontier language models (GPT-4o, GPT-5, Gemini 2.5 Pro, Gemini 2.5 Flash,
and Claude Haiku 4.5, Sonnet 4.6, and Opus 4.6), each run three times, on two engineering
assignments graded independently by three human graders each: a 133-point Operating Systems
assignment (40 students, 240 question instances) and a 10-point Biomaterials assignment
(5 students, 25 question instances). That yields 945 model gradings set against 755 human
question-level judgements.

We propose and apply a human-ceiling protocol. Quantify inter-human reliability first, build
a consensus reference from it, then ask whether inserting the model as an additional rater
preserves the panel's reliability. Three findings follow. First, the protocol earns its keep:
on the Operating Systems assignment a conventional single-reference analysis reverses the
sign of measured grading bias for five of seven models, and reports significant bias
(p < 0.001) for two models that the corrected analysis cannot distinguish from the human
consensus. Second, no model reaches human variability in either course. The best model's
disagreement with the human panel runs 2.80x the humans' disagreement with each other on
Operating Systems and 1.76x on Biomaterials, and inserting any model as a fourth rater lowers
ICC(2,1) in both courses (by 0.091 to 0.209, and by 0.077 to 0.161, respectively). Third, the
error is mostly systematic. Twelve of fourteen model-course pairs show statistically
significant harshness, and that offset accounts for most of the gap, which makes it
correctable.

Our two courses differ sharply in human ceiling (ICC 0.956 against 0.504) while sharing one
protocol and one model set, so we reproduce the literature's contradictory verdicts inside a
single study. We argue that much of the field's disagreement concerns reference quality
instead of model capability.

Current capability does not justify autonomous LLM grading. It does justify per-rubric bias
calibration paired with human review of flagged items. We ask the field to report
inter-human reliability as a precondition for any claim about LLM grading accuracy.

**Keywords.** automated grading, large language models, inter-rater reliability, assessment,
engineering education, LLM evaluation

---

## 1. Introduction

Grading open-response engineering work is expensive, slow, and inconsistent in ways any
instructor who has moderated a teaching team will recognise. Capable large language models
(LLMs) have prompted a wave of proposals to automate it.

The published results disagree with each other. Kortemeyer reports R² = 0.84 against human
graders on introductory physics derivations [4]. Bernik et al. report Pearson 0.91 on
programming assignments [6]. Gobrecht et al. report an automated grader whose median absolute
error came in 44% below that of certified human re-graders [3], and Tang et al. report
human-AI agreement on physics exams comparable to human inter-rater reliability [9]. Set
against those, Mathew et al. report every tested model below QWK 0.30 where human raters
reach 0.72 [10], Caraeni et al. judge GPT-4o's accuracy on handwritten mathematics "too low
for real-world settings" [11], and Lundgren finds low inter-rater reliability between GPT-4
and instructors even where average grades match [12]. These are opposite answers from
competent studies.

Most of this evidence shares a design choice that we believe explains much of the spread. A
model's output is compared against one reference, whether a single instructor, a single
teaching assistant, or an answer key, and agreement with that reference gets reported as
accuracy. The design is convenient, and for objective item types it is sound. For
rubric-scored open response it carries an assumption that is rarely tested: that the
reference grader's mark *is* the correct grade, and by extension that a second human grader
would have produced much the same number.

Flodén states the resolving hypothesis without being able to test it. Having found that
ChatGPT's grades differed from the teachers' in most cases, he observes that "it is not
unlikely that two different human graders could result in similar discrepancies" [5]. Testing
that needs several independent human graders on the same items, which his study, like most,
did not have.

The assumption is measurable, and measuring it shows it does not hold uniformly. Across our
two courses the same three-grader protocol produced inter-rater ICC(2,1) of 0.956 in one and
0.504 in the other. In the first, human graders are close to interchangeable and a demanding
bar for an LLM makes sense. In the second the humans disagree substantially among themselves,
so a model judged against any one of them would be scored largely on the accident of which
human it was compared against. One number, "correlation with the instructor", cannot separate
these two situations, and they call for opposite deployment decisions.

This paper makes four contributions.

**A method.** We describe a human-ceiling protocol for evaluating LLM graders (Section 3). It
requires measuring inter-human reliability before any model is scored, building a consensus
reference rather than one privileged grader, and applying an insertion test: does adding the
model to the human panel as an additional rater preserve the panel's reliability? The protocol
converts a vague question, "is the model accurate?", into an answerable one. Is the model
inside the variability the department already tolerates among its own graders?

**Evidence.** We apply the protocol to seven frontier models across two dissimilar engineering
assignments, with three independent human graders per assignment and three repeated runs per
model (Sections 4 to 6). Repeated runs separate a model's systematic error from its
run-to-run instability, which single-run studies cannot see and which matters for deployment.

**Guidance.** The dominant error component is a systematic harshness offset instead of noise,
and from that we derive deployment recommendations that land between "replace the TAs" and
"do not use LLMs" (Section 8).

**A reconciliation.** Our two courses have very different human ceilings while sharing one
protocol and one set of models, so we can reproduce the field's contradictory verdicts inside
a single study (Section 8.5). We argue that much of the disagreement in the literature
concerns reference quality instead of model capability, and we state the prediction that
would test this.

We also report what the conventional analysis would have concluded on the same data
(Section 5.7). The difference is substantial. Five of seven models change the sign of their
reported bias, and the ranking of which models count as "significantly biased" reorders. We
present this to argue that the human ceiling is a precondition for interpreting any of these
numbers, not to criticise prior work.

---

## 2. Related Work

### 2.1 Automated grading before LLMs

Automated assessment has a long history in engineering and computing education. Burrows,
Gurevych, and Stein's survey of automatic short answer grading (ASAG) identifies 35 systems
across five temporal eras, tracing the field from concept-mapping and information-extraction
approaches through corpus-based and machine-learning methods [1]. Their component analysis
established the field's most durable finding, that performance depends heavily on the item,
and argued that an era of evaluation, not of new architectures, is what consolidates the
field. Our results in Section 5.6, where per-question MAE varies by a factor of five across
items inside a single assignment, agree with this.

Fine-tuned transformers set the pre-LLM ceiling. Dada et al. released a structured dataset of
open-ended university responses and found Longformer strongest among traditional and
transformer approaches, at Spearman 0.77 against human graders [2]. Gobrecht et al. matter
most for our purposes. They trained an ASAG model on university exam data and evaluated it
against certified human domain experts re-grading historic exams, reporting that the model's
median absolute error came in 44% below the human re-graders', and concluding that automated
grading is more consistent and can therefore increase fairness [3]. That is the strongest
pro-automation claim in the literature we review, and it is explicitly ceiling-referenced. We
return to it in Section 8.5.

### 2.2 LLMs as graders in higher education

Instruction-tuned LLMs removed the need for per-item training data and set off a rapid wave of
evaluations in higher education. The reported results contradict each other, and that
contradiction is where this paper starts.

**Favourable findings.** Kortemeyer graded handwritten introductory-physics derivations with
MathPix and GPT-4, reaching R² = 0.84 against human graders, but concluded that the workflow
suits formative feedback and that for final evaluations it "would best be used to assist human
graders" [4]. Flodén ran the study closest in design to ours: three Master's-level exams, 463
responses, each graded at least three times by ChatGPT for 1,389 total gradings, compared
against the teachers who set them [5]. Seventy per cent of gradings fell within 10% of the
teachers' marks and 31% within 5%. Bernik et al. compared GPT-4-turbo, Claude 3, and Gemini
1.5 Pro on 315 Python assignments, with GPT-4 reaching Pearson 0.91 and mean absolute
deviation of 0.68 points [6]. Poličar et al. concluded that with well-designed prompts, LLM
grading of a bioinformatics course matched human teaching assistants on both scoring and
feedback [7]. Henkel et al. reported GPT-4 at Cohen's kappa 0.70 against human raters at 0.75
on K-12 short-answer marking [8]. Tang, Ambrose, and Cheng scored constructed-response physics
exams using four instructors across two rounds and found human-AI agreement on total scores
comparable to human inter-rater reliability, though the model struggled on mid-range responses
involving partial or ambiguous reasoning [9].

**Unfavourable findings.** Mathew et al. evaluated six models on the ASAP and DREsS essay
corpora and found all of them below QWK 0.30 against a human-human ceiling of QWK 0.72,
reporting that LLMs compress scoring ranges, inflating underdeveloped essays while penalising
minor language errors in otherwise strong work [10]. Caraeni, Scarlatos, and Lan evaluated
GPT-4o on handwritten university mathematics exams and found that rubrics improved alignment
while overall accuracy stayed "too low for real-world settings" [11]. Lundgren found that
GPT-4 produced comparable average grades to human instructors while its inter-rater
reliability with those instructors stayed low, and that it graded risk-aversely, attending to
surface features over disciplinary standards [12].

Broader reviews of AI-assisted grading document the efficiency case alongside a persistent
need for human oversight [13], and instructor acceptance remains an independent constraint on
adoption [14].

Documented failure modes include systematic leniency or severity [10], [12], compression
toward the centre of the scale [5], [10], sensitivity to prompt formulation, where
semantically equivalent prompts shift behaviour substantially [15], and instability across
repeated invocations of an identical prompt. Stureborg et al. characterise LLM evaluators as
inconsistent and biased, reporting skewed rating distributions, anchoring effects, and low
agreement with themselves on identical samples [16]. That last property motivates our
repeated-run design in Section 5.6. Single-run studies cannot see it.

### 2.3 LLM-as-judge and its evaluation designs

A parallel NLP literature evaluates LLMs as judges of model output rather than student work.
Zheng et al. introduced MT-Bench and Chatbot Arena and documented position, verbosity, and
self-enhancement biases in LLM judges [17]. They also benchmark GPT-4's agreement with humans
(roughly 85%) against human-human agreement (roughly 81%), which is exactly the
ceiling-referenced comparison we argue educational studies should adopt as standard. The
insertion test of Section 3 comes out of this tradition of treating the judge as one rater
among several rather than as an oracle.

### 2.4 Inter-rater reliability in educational assessment

Measurement theory has long held that a single rater is an unreliable instrument. Shrout and
Fleiss formalised the intraclass correlation coefficient and its six forms, separating the
reliability of a single rater from that of a k-rater average [18]. Cohen's weighted kappa
extends chance-corrected agreement to ordinal scales with partial credit [19]. Hayes and
Krippendorff argue for alpha as a standard reliability measure partly because it tolerates
missing judgements [20], which real grading data has, as Section 5.1 demonstrates.
Generalizability theory, developed by Cronbach et al. [21] and applied to performance
assessment by Brennan [22], decomposes measurement error into facets including raters, items,
and occasions, and is the natural framework for the question we ask.

### 2.5 The gap

Ceiling-referenced comparison is not our invention, and we make no claim on it. Gobrecht et
al. [3], Henkel et al. [8], Tang et al. [9], Mathew et al. [10], and Zheng et al. [17] all
report human agreement alongside model agreement. Our contribution is narrower.

Consider what the literature above collectively asserts. Gobrecht et al. report a model
beating human re-graders by 44% [3]. Tang et al. report parity with human inter-rater
reliability [9]. Mathew et al. report QWK below 0.30 against a human 0.72 [10]. Those are
opposite answers to one question, produced by competent studies using defensible statistics.

Flodén states the resolving hypothesis explicitly and cannot test it, for want of a second
independent grader on the same scripts [5]. The hypothesis has been sitting there for years
unmeasured.

Our contribution is therefore threefold.

1. **The insertion test.** Reporting a human ceiling beside a model score is weaker than
   asking whether the model, added to the existing panel, preserves that panel's reliability.
   We are not aware of this test being applied to LLM grading of engineering coursework, and
   it is the question a department faces when it considers deployment.
2. **A demonstration, on one dataset, that the design choice changes the answer.** Section 5.7
   shows that the conventional single-reference, run-pooled analysis reverses the sign of
   measured bias for five of seven models on our data, and reverses significance verdicts in
   both directions. That converts "you should measure the ceiling" from methodological advice
   into an empirical result.
3. **Two courses, one protocol, ceilings differing by a factor of nearly two.** This lets us
   address the contradiction above directly (Section 8.5) instead of adding one more data
   point to it.

---

## 3. The Human-Ceiling Protocol

The protocol has six steps. Steps 1 and 2 concern the humans and must finish before any model
is scored. Steps 3 to 6 concern the models.

### Step 1. Quantify the human ceiling

With three or more independent graders scoring the same items, compute at the
*question-instance* level:

- **ICC(2,1)**, the reliability of a single randomly chosen rater, which is the quantity that
  matters when one grader will grade one submission.
- **ICC(2,k)**, the reliability of the k-rater average, which is the quality of the consensus
  itself.
- **Krippendorff's alpha** (interval), which tolerates the missing judgements real grading
  data carries.
- **Kendall's W**, the concordance of the rank ordering graders impose on students.
- **Mean pairwise human-human MAE** in points per question, the interpretable form, and the
  number instructors actually reason about.

Together these define the ceiling. The last one serves the deployment question best, because
it answers a question a department can act on: how many points apart are two of our own
graders, on average?

### Step 2. Form a consensus reference

In place of privileging one grader, take the per-question mean across graders as the
reference, with the median computed as a sensitivity check. Where the two diverge, the item is
contested and should be flagged instead of averaged away. Missing judgements get skipped in
the mean, never imputed, but the resulting asymmetry has to be tracked, because it propagates
(Section 5.1).

### Step 3. Score each model against the consensus

Compute MAE, RMSE, Pearson r, Spearman rho, agreement-R², bias (signed mean error), and error
SD. Compute these per run and then average across runs without pooling them, so that
repeated measurements of the same item do not enter the analysis as independent observations.

### Step 4. The insertion test

Insert the model into the panel as an additional rater and recompute ICC(2,1) and
Krippendorff's alpha over the enlarged panel, then compare to the human-only baseline. Where
reliability holds, the model is contributing rater-quality judgements. Where it drops, the
model is injecting disagreement. This is the protocol's central test, because it asks the
deployment question directly. Would we accept this as one more grader on the team?

### Step 5. The within-human-variability test

Compute each model's mean absolute disagreement with each individual human, then compare it
to the mean human-human disagreement. A model whose disagreement with the humans is no larger
than the humans' disagreement with each other is an acceptable grader by the department's own
operative standard. This yields the interpretable ceiling ratio:

> ceiling ratio = (model's mean MAE vs. individual humans) / (mean human-human MAE)

A ratio of 1.0 puts the model exactly as far from the humans as they are from each other.
Anything above 1.0 is a model that disagrees with the staff more than the staff disagree
among themselves.

### Step 6. Run-to-run self-consistency

Human graders grade once. A model can be invoked repeatedly. For every (student, question,
model) triple, compute the standard deviation of scores across runs and average. This measures
grading noise, a failure mode distinct from bias, invisible to run-pooled analysis, and
directly relevant to fairness, because a noisy grader assigns different grades to identical
work.

Order matters here. Steps 1 and 2 must precede steps 3 to 6. Once a single grader has been
designated "the reference", every subsequent number inherits that grader's idiosyncrasy, and
the ceiling cannot be recovered after the fact.

---

## 4. Study Design

### 4.1 Grading pipeline

All models ran through one grading service that presents each model with an identical task
structure. For each question it supplies the question text, an optional reference answer, the
rubric, the maximum points, and the student's answer, then requests a per-question score with
strengths, areas for improvement, and a breakdown. Questions are flattened, so multi-part
questions expand into their sub-parts, and each submission is graded in a single bulk call
instead of one call per question, which lets the model see the whole submission in context.

Objectively-scorable item types (multiple choice, true/false) are graded deterministically by
the service, never by the model, and are excluded from every analysis reported here. Every
number in this paper therefore concerns open-response, rubric-scored items.

Three provider APIs were used: the OpenAI Chat Completions API, Amazon Bedrock's Converse API
for the Anthropic models, and the Google GenAI API for the Gemini models.

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

Each model graded every submission three times. These decoding settings are not fully uniform
across providers. We treat that as a limitation, not a design feature; see Section 9.

### 4.3 Course 1: Operating Systems (primary study)

An undergraduate Operating Systems assignment with 6 open-response questions, 133 points
total, and per-question maxima ranging from 15 to 40 points. Forty student submissions were
graded independently by three teaching assistants (TA1, TA2, TA3), and by each of the seven
models three times. Student answers reached the models as structured text.

That yields 240 question instances, 680 human question-level judgements (720 less the 40
missing TA2 Q6 scores, Section 5.1), and 5,040 model question-level scores
(240 x 7 models x 3 runs).

### 4.4 Course 2: Biomaterials (contrast study)

A Biomaterials assignment with 5 open-response questions at 2 points each, 10 points total.
Five student submissions were graded independently by three human graders, and by each of the
seven models three times. Submissions reached the models as PDFs ingested directly, in place
of extracted structured text.

That yields 25 question instances, 75 human judgements, and 525 model scores.

**On the role of this study.** With five students this arm is underpowered, and we do not
treat it as independent confirmation of the Operating Systems results; its confidence
intervals are wide and overlapping (Section 6). It earns its place because its human ceiling
is low, ICC(2,1) = 0.504 against 0.956 for Operating Systems, which makes it the more
informative case for the protocol itself. The bar an LLM must clear is a property of the
course and its graders, never a universal constant, and this study shows that directly.

### 4.5 Analysis

Both courses were analysed with the identical protocol of Section 3. Confidence intervals are
bootstrap (percentile) intervals. Paired comparisons between a model and the human consensus
use the run-averaged score per student as the unit, giving n = 40 and n = 5 respectively, and
are reported with paired t-tests and Wilcoxon signed-rank tests alongside Cohen's d. Rank
correlations across the two courses are Spearman's rho over the seven models.

---

## 5. Study 1: Operating Systems

### 5.1 A data-quality finding that changes the results

One property of the human data has to be stated before any agreement figure, because it
propagates into every naive comparison.

TA2 did not grade Question 6. All forty of TA2's Q6 scores are missing, so TA2's recorded
total sums Q1 to Q5 (out of 93) while TA1's and TA3's sum Q1 to Q6 (out of 133). Any analysis
that averages the three recorded totals to form a reference produces a reference roughly 8.8
points too low, because one third of the average is missing an entire question worth 40
points.

The effect is to inflate every model's apparent generosity by roughly 8.8 points. Section 5.7 shows
this is enough to reverse the sign of the reported bias for five of the seven models. We
handle it by computing the consensus per question instance, so the Q6 consensus is the TA1/TA3
mean, by computing ICC and Kendall's W on the fully-crossed Q1 to Q5 subset, and by using
Krippendorff's alpha, which tolerates missingness, on all 240 instances.

We report this in detail because it is an ordinary failure. Partial grading, split grading
duties, and missing rubric rows are routine features of real course data, and a
single-reference pipeline absorbs them silently.

### 5.2 The human ceiling

**Table 1. Inter-human agreement, Operating Systems.**

| Measure | Value |
|---|---|
| ICC(2,1), single rater (200 fully-crossed units) | **0.956** |
| ICC(2,k), 3-TA average | 0.985 |
| Krippendorff's alpha, interval (240 units) | 0.956 |
| Kendall's W | 0.969 |
| Mean pairwise TA-TA MAE | **1.113 pts/question** |
| TA1–TA2 / TA1–TA3 / TA2–TA3 | 1.10 / 1.53 / 0.70 |
| Mean pairwise quadratic-weighted kappa (grade bins, Q1–Q5) | 0.896 |
| Mean pairwise TA-TA MAE, total score (Q1–Q5, /93) | 3.46 pts |
| TA mean totals (Q1–Q5, /93) | TA1 56.5, TA2 55.6, TA3 53.9 |

These TAs are close to interchangeable. An ICC(2,1) of 0.956 means a single randomly chosen TA
already reproduces the panel almost exactly, and the three TAs' mean totals span 2.6 points out
of 93. The bar for a model in this course is high, and it is high for a principled reason
rather than a convention that ICC should exceed 0.9.

### 5.3 Models against the consensus

The mean and median consensus agree closely (r = 0.998), and we use the mean.

**Table 2. Question-level metrics vs. the TA consensus (240 instances, per-run averaged).
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

Read on its own the table looks like a success. Correlations of 0.77 to 0.88 sit in the range
routinely offered as evidence that LLMs can grade. Then read the MAE column against the
human ceiling. The best model lands 3.148 points per question from the consensus against a
human-human disagreement of 1.113, nearly three times as far. Correlation runs high because
the models rank students correctly. The absolute grades are not close.

Every bias is negative. All seven models grade this assignment harshly.

*[Figure 1: `Grading_Dataset_OS/outputs/4_2_llm_vs_ta_scatter.png`. Per-model scatter of
model score against TA consensus.]*

*[Figure 2: `Grading_Dataset_OS/outputs/5_4_bland_altman.png`. Bland–Altman agreement. The
mean-difference lines sit below zero across the score range, which confirms a systematic
offset and not a scale-dependent distortion.]*

### 5.4 The insertion test

**Table 3. ICC(2,1) with each model inserted as a fourth rater. Human-only baseline:
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
costs 0.091 of ICC, and adding the weakest costs 0.209. Put in the language a department would
use, every one of these models is a worse-than-average member of this grading team, and
swapping a TA for any of them measurably degrades the consistency of the grading students
receive.

*[Figure 3: `docs/paper/figures/fig3_icc_as_fourth_rater_os.png`]*

### 5.5 The within-human-variability test

**Table 4. Model disagreement with each individual TA, points per question.
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

Zero of seven models fall within human variability, and the shortfall is large. The best model
would have to cut its disagreement by 64% to reach the ceiling.

*[Figure 4: `Grading_Dataset_OS/outputs/9_5_within_human_variability.png`]*

### 5.6 Where the models fail, and how noisily

Per-question analysis locates the error. Q6, the 40-point question and the one TA2 left
ungraded, is hardest for every model: Claude Haiku's MAE on Q6 reaches 7.76 points against
1.47 on Q4. Q4, the most constrained item, is where every model does best, at MAE 1.31 to
1.58. The pattern matches the pre-LLM autograding literature, where constrained items automate
well and open-ended synthesis does not.

*[Figure 5: `Grading_Dataset_OS/outputs/8_2_per_question_mae_heatmap.png`]*

**Table 5. Run-to-run standard deviation (points per question, mean over all
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

This is a fairness result as much as a reliability one. Gemini 2.5 Flash, invoked twice on
identical work, produces scores differing by 1.58 points per question on average, which
compounds to 9.5 points across a six-question assignment. The two least noisy models also lead
on accuracy, though the orderings diverge: GPT-4o sits mid-pack on noise while finishing last
on accuracy, and Gemini 2.5 Pro places third on accuracy while ranking second-noisiest.

*[Figure 6: `Grading_Dataset_OS/outputs/9_6_run_to_run_noise.png`]*

### 5.7 What the conventional analysis would have concluded

We now run the analysis the conventional way on the same data, taking the reference as the
mean of the three recorded TA totals, which inherits the TA2/Q6 gap, and treating each
(student, run) row as an independent observation, giving n = 120 where the corrected
analysis uses n = 40.

**Table 6. Naive vs. protocol-corrected bias, Operating Systems. Negative bias = harsher
than the human consensus.**

| Model | Naive bias | Corrected bias | Sign flip? | Naive p | Corrected p (n=40) | Cohen's d | Corrected verdict |
|---|---|---|---|---|---|---|---|
| Claude Haiku | −4.70 | −13.52 | no | 0.001 | <0.0001 | −0.908 | Significantly harsh |
| GPT-4o | −0.75 | −9.57 | no | **0.667** | **0.0013** | −0.548 | Significantly harsh |
| Claude Sonnet | **+2.35** | −6.46 | **Yes** | 0.047 | 0.0038 | −0.530 | Significantly harsh |
| GPT-5 | **+3.39** | −5.42 | **Yes** | 0.023 | 0.0366 | −0.359 | Significantly harsh |
| Claude Opus | **+4.50** | −4.31 | **Yes** | <0.001 | 0.0467 | −0.366 | Significantly harsh |
| Gemini 2.5 Flash | **+6.35** | −2.47 | **Yes** | **<0.001** | **0.2162** | −0.195 | Not significant |
| Gemini 2.5 Pro | **+6.76** | −2.06 | **Yes** | **<0.001** | **0.3011** | −0.166 | Not significant |

The conventional analysis of this dataset would have supported three claims, all artefacts.

1. **That five of seven models grade generously.** They do not. All seven grade harshly, and
   the sign reversal traces entirely to the missing-TA2-Q6 reference contamination.
2. **That Gemini 2.5 Pro and Flash exhibit highly significant bias (p < 0.001).** Corrected,
   these two are the models whose bias cannot be statistically distinguished from zero
   (p = 0.30 and p = 0.22). They are the best-calibrated models in the study.
3. **That GPT-4o is unbiased (p = 0.667).** Corrected, GPT-4o grades significantly harshly
   (p = 0.0013, d = −0.548).

Two distinct errors produce this. Reference contamination shifts every bias estimate by a
constant. Pseudoreplication, which treats three runs of the same student as three independent
observations, inflates n threefold and shrinks p-values, manufacturing significance for effects
that are small relative to between-student variance.

Neither error is exotic, and neither is visible from inside the naive analysis. Only the step-1
requirement to characterise the human panel before scoring any model surfaces the first, and
only run-averaging surfaces the second.

---

## 6. Study 2: Biomaterials

### 6.1 A different ceiling

**Table 7. Inter-human agreement, Biomaterials, with Operating Systems for comparison.**

| Measure | Biomaterials | Operating Systems |
|---|---|---|
| ICC(2,1), single rater | **0.504** [0.220, 0.730] | 0.956 |
| ICC(2,k) | 0.753 | 0.985 |
| Krippendorff's alpha | 0.469 | 0.956 |
| Kendall's W | 0.505 | 0.969 |
| Mean pairwise weighted kappa | 0.516 | 0.896 |
| Mean human-human MAE | **0.227 pts/q** [0.133, 0.320] | 1.113 pts/q |
| Human mean totals (/10) | H1 9.30, H2 7.80, H3 9.10 | n/a |

The same three-grader protocol, applied to a different course, yields a different instrument
altogether. An ICC(2,1) of 0.504 is moderate at best, and the graders' mean totals span 1.5
points out of 10, with H2 grading markedly harder than H1 and H3.

Two factors contribute. The rubric is coarse, at five questions of 2 points each, with most
human scores landing in [1.5, 2.0]. Range restriction of that kind mechanically depresses ICC,
which is a ratio of between-subject to total variance, so when students genuinely differ
little even small rater disagreements dominate. It also drives the negative R² values in Table
8: with almost no variance around the consensus to explain, any deviation makes a model worse
than predicting the mean. Under range restriction, R² is the wrong summary and MAE is the
right one.

Had we evaluated a model against H2 alone it would look accurate, and against H1, harsh. The
ceiling framework makes that ambiguity explicit where the choice of a single reference grader
would bury it.

### 6.2 Models against the consensus

**Table 8. Question-level metrics vs. consensus (25 instances, 2 points max per question).
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

Every confidence interval on bias excludes zero. All seven models grade significantly harshly,
with paired t and Wilcoxon both at p ≤ 0.001 and large effect sizes (|d| = 0.81 to 1.86). On a
10-point assignment the human graders average 8.7 while the models average 6.1 to 7.4, a gap
of well over a letter grade.

The MAE intervals overlap heavily. With 25 question instances the ordering of adjacent models
here is not statistically resolved, and we make no claim that it is. The direction is
resolved: every model, harsh, with a large effect size.

*[Figure 7: `Biomaterials/outputs/04_bias_by_model.png`]*

*[Figure 8: `Biomaterials/outputs/10g_radar.png`]*

### 6.3 Insertion, variability, and noise

The human-only baseline is ICC(2,1) = 0.504. With a model inserted as a fourth grader the
values run: Gemini 2.5 Flash 0.426, Claude Opus 0.425, Claude Sonnet 0.408, Gemini 2.5 Pro
0.407, Claude Haiku 0.397, GPT-5 0.347, GPT-4o 0.342, for drops of 0.077 to 0.161.

The verdict matches Operating Systems. No model preserves panel reliability, even against a
panel that is itself only moderately reliable.

*[Figure 9: `docs/paper/figures/fig9_icc_as_fourth_rater_bio.png`]*

On the within-human test, model MAE against individual humans ranges from 0.399 (Claude Haiku)
to 0.554 (GPT-5) against a ceiling of 0.227, giving ceiling ratios of 1.76x to 2.44x. Again,
zero of seven within human variability.

Run-to-run SD ranges from 0.066 (Claude Opus) to 0.208 (GPT-4o) points per question on a
2-point scale, proportionally 3.3% to 10.4% of the item's value. Model ordering by noise
resembles Operating Systems, with the Claude models most stable and GPT-4o and Gemini 2.5
Flash least.

### 6.4 One instructive item

Per-question analysis flags Q4 as an interesting failure. The humans agreed on it more than on
any other item (SD 0.115) and the consensus came to 1.9/2.0, yet the models' mean absolute
error reached 0.444, their second-worst. An item unambiguous to every human grader is one the
models systematically misread. On Q5 the humans were most split (SD 0.273, consensus 1.5)
while the models were confidently wrong in a consistent direction.

Both patterns argue against using model-human disagreement as an automatic flag for a
contested item. The two do not track each other.

---

## 7. Cross-Domain Synthesis

### 7.1 What replicates, stated precisely

The two courses differ in discipline, rubric granularity (2-point against 40-point items),
scale (10 against 133 points), cohort size, input modality (PDF against structured text), and
human ceiling (0.504 against 0.956). Against that spread, we ask what survives.

**Rank correlation across the two courses (Spearman's rho, n = 7 models):**

| Metric | rho | p |
|---|---|---|
| Composite score | 0.750 | 0.052 |
| ICC as 4th rater | 0.643 | 0.119 |
| Run-to-run SD | 0.679 | 0.094 |
| MAE | 0.571 | 0.180 |

The full ranking does not replicate. None of these correlations reaches significance at n = 7,
and we explicitly decline the claim that our model ordering generalises. Reporting it as a
stable leaderboard would go beyond what this data supports.

What does replicate:

1. **The top two, exactly.** Claude Opus ranks 1 and Claude Sonnet ranks 2 in both courses, on
   every individual metric we computed.
2. **The OpenAI models sit bottom-tier in both**, though they trade places, with GPT-4o last
   on Operating Systems and GPT-5 last on Biomaterials.
3. **The middle does not order consistently.** Gemini 2.5 Pro places 3rd on Operating Systems
   and 5th on Biomaterials; Claude Haiku places 6th and 4th respectively.
4. **Both verdicts replicate without qualification.** Zero of seven models within human
   variability, in both courses. Every model harsh, in both courses, with twelve of fourteen
   model-course pairs significantly so, the exceptions being Gemini 2.5 Pro and Flash on
   Operating Systems.

### 7.2 The ceiling ratio

**Table 9. Ceiling ratios: model disagreement with humans, in units of human-human
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

Every bar in both courses sits above 1.0. The Biomaterials ratios come out lower because the
humans there grade less consistently, which lowers the bar; the models do not grade that
course better in absolute terms. Here is the framework's central point rendered numerically.
The same model can be 2.80x or 1.80x from acceptable depending entirely on whose grading it is
asked to match.

A practical corollary follows. The courses where an LLM grader is most likely to clear the
human bar are the courses where human grading is least reliable, which is a much weaker
endorsement than "the LLM is accurate" and should be described that way when deploying.

### 7.3 Bias dominates variance

Across both courses the signed bias accounts for most of the mean absolute error. On
Biomaterials, mean bias runs −0.40 points per question against a mean MAE of 0.43, so the
models are almost entirely offset, with comparatively little residual scatter. On Operating
Systems the offset takes a smaller share of the total while staying statistically significant
for five of seven models.

This is the most actionable finding in the paper, and Section 8.1 develops it.

### 7.4 Latency

**Table 10. Grading latency per submission (seconds).**

| Model | OS (median) | Biomaterials (mean) |
|---|---|---|
| GPT-4o | **8.2** | **7.4** |
| Gemini 2.5 Flash | 25.9 | 28.5 |
| Claude Haiku | 27.0 | 20.0 |
| Gemini 2.5 Pro | 37.6 | 31.3 |
| Claude Opus | 52.6 | 37.9 |
| Claude Sonnet | 54.2 | 38.8 |
| GPT-5 | 134.2 | 88.0 |

Latency spans a factor of 16 and runs inversely to accuracy at both extremes, since the
fastest model is the least accurate and the slowest (GPT-5) sits mid-pack. Claude Opus
delivers the best accuracy at roughly 40% of GPT-5's latency. On a 40-student assignment even
the slowest configuration finishes a full grading pass in under 90 minutes unattended, so
latency is unlikely to bind in practice.

*[Figure 11: `Grading_Dataset_OS/outputs/4_6_latency_boxplot.png`]*

---

## 8. Discussion

### 8.1 The error is an offset, and offsets are correctable

Random error and systematic error carry very different practical consequences, which is what
makes Section 7.3 matter. Improving a grader that is wrong at random means improving the
grader itself. Consistent wrongness in one direction reduces to a calibration problem, and
calibration is cheap.

On Biomaterials, subtracting each model's mean bias from its scores would cut mean absolute
error from roughly 0.43 toward the residual error SD. The models already rank students
reasonably, and the harshness offset is what pushes the absolute grades out of acceptable
range. A per-rubric calibration constant, estimated from a modest sample of double-graded
submissions, addresses that directly.

The constraints deserve emphasis. The offset attaches to the rubric, not the model: Claude
Haiku's bias runs −2.253 points per question on Operating Systems and −0.265 on Biomaterials,
and no rescaling converts one into the other. Calibration therefore needs human-graded
submissions from the same assignment, which reintroduces the human effort automation was meant
to remove, though at a sample size far below full grading. Calibration also cannot repair the
run-to-run noise of Section 5.6, which is a separate defect.

### 8.2 What we do not recommend

**Autonomous grading of record.** This data does not support it in either course. Zero of
fourteen model-course pairs fall within human variability, every model degrades panel
reliability on insertion, and the shortfall is wide.

**Substituting a model for a TA on a grading team.** The insertion test evaluates exactly
this, and every model fails it.

**Trusting a high correlation.** Table 2 is the cautionary case, with r = 0.879 sitting
alongside an MAE 2.80x the human ceiling. Correlation measures whether the model ranks
students correctly and is nearly blind to a uniform offset, so a model can correlate at 0.88
with the TAs while awarding every student a grade one band too low. Any evaluation reporting
correlation without absolute error, referenced to human-human disagreement, under-reports.

### 8.3 What the data does support

**Second-opinion flagging.** Use the model as an additional, non-authoritative rater and
surface the submissions where model and human diverge most. Correctness is not the
requirement here; the model needs to be uncorrelated enough with the human's error to catch
slips. Section 6.4 is a caution, since model-human divergence did not track item
contentiousness, so this wants validating per course.

**Triage for grading order.** Model scores correlate 0.77 to 0.88 with the consensus, ample
for ordering a grading queue so borderline submissions reach a human first.

**Draft feedback.** The models produce per-question strengths, weaknesses, and breakdowns.
Nothing in our data speaks to the quality of that prose, since we evaluated scores only. The
grading task and the feedback task fail differently, and a model harsh by a consistent offset
may still write useful formative comments. This needs separate evaluation.

**Rubric quality diagnosis.** Running step 1 of the protocol is what surfaced the Biomaterials
ceiling of 0.504. That finding has independent value to the instructor regardless of any LLM,
since it says the rubric does not discriminate reliably between graders. Running the human half
of this protocol is worthwhile even where no model is ever deployed.

### 8.4 Model selection, if deploying

Claude Opus 4.6 and Claude Sonnet 4.6 lead on every metric in both courses and are the two most
run-stable models, with Sonnet reaching near-identical accuracy to Opus at comparable latency.
GPT-4o finished last or near-last on accuracy in both courses, and its speed advantage does not
compensate. Gemini 2.5 Pro and Flash were the best-calibrated models on Operating Systems, the
only two with statistically non-significant bias, while also being the noisiest across runs.
That combination suits ensemble averaging, where repeated sampling suppresses noise and good
calibration survives. We did not test ensembling. Among the configurations our results point
to, it looks the most promising and remains untested.

### 8.5 Reconciling the contradictory literature

Section 2.2 laid out an unresolved disagreement. Gobrecht et al. report an automated grader
beating human re-graders by 44% on median absolute error [3] and Tang et al. report parity with
human inter-rater reliability [9], while Mathew et al. report QWK below 0.30 against a human
ceiling of 0.72 [10] and Caraeni et al. judge accuracy too low for deployment [11].

Our two studies reproduce that disagreement inside a single paper, using one protocol and one
set of models. Ranked against the Operating Systems TAs, the best model sits 2.80x outside
human variability and looks clearly unfit. Ranked against the Biomaterials graders, the same
model sits 1.80x outside, still failing, but by a margin a slightly different rubric or a
slightly noisier panel would erase. The panel changed between those two verdicts while the
models stayed fixed.

That suggests a specific reading of the literature. Studies reporting parity or better tend to
involve reference conditions with high human variability, such as re-grading historic exams
without the original grading context [3], or rubrics on which instructors themselves diverge.
Studies reporting failure tend to involve reference conditions with well-controlled human
agreement, such as essay corpora with trained, calibrated raters reaching QWK 0.72 [10]. The
apparent disagreement about model capability may be substantially a disagreement about
reference quality.

We advance this as an explanation consistent with our data, never as a demonstrated one. We
cannot recompute other authors' ceilings, and we do not claim their conclusions are wrong. The
testable prediction is that reported LLM-human agreement should correlate negatively with the
inter-human reliability of each study's reference panel. A meta-analysis could check that, and
we suggest it as future work.

The practical implication is uncomfortable, and it restates Section 7.2. An LLM grader is most
likely to clear the human bar precisely where human grading is least reliable. "The model
performs as well as our graders" and "our graders do not agree with each other" are compatible
statements, and reporting practice tends to surface only the first.

---

## 9. Threats to Validity

**The prompt is not identical across providers.** The grading service emits a compact
delimiter-based format for Gemini models, including an explicit instruction to keep each field
under 30 words, and a JSON format for OpenAI and Anthropic models. Gemini's results are
therefore confounded with a different prompt and a tighter output-length constraint. Gemini's
cross-model comparisons should be read with that in mind. The within-Gemini findings on
calibration and noise are unaffected.

**Decoding settings are not uniform.** Six models ran at temperature 0.1 while GPT-5 ran with
high reasoning effort and no temperature control. GPT-5's run-to-run SD was therefore not
measured under the same conditions as the others, and its noise figure is not directly
comparable.

**Token budgets differ.** Anthropic and Gemini models ran at 2000 max tokens against 16384 for
non-reasoning OpenAI models, which could truncate long rubric feedback asymmetrically.

**Input modality is confounded with course.** Operating Systems answers arrived as structured
text and Biomaterials submissions as directly-ingested PDFs, so any cross-course difference may
be a modality effect and not a course effect. The within-course results, which carry all
our primary claims, are unaffected.

**The Biomaterials study is underpowered.** Five students, 25 question instances, and wide
overlapping confidence intervals. We treat it as a contrast case for the protocol, never as
independent replication, and we make no claim that its model ordering is resolved.

**The consensus is a reference, not ground truth.** Averaging three graders does not produce
correctness. Where the humans agree at ICC 0.504 the consensus is itself an unreliable target,
and a model penalised against it may well be right.

**Range restriction on Biomaterials.** Five 2-point items with most scores in [1.5, 2.0]
mechanically depresses ICC and produces negative R². We report MAE alongside for that reason,
and the ICC comparison across the two courses should not be read as a pure difference in
grader skill.

**One assignment per course, one institution.** Both studies draw on single assignments from a
single institution with one rubric each. Question-level findings (Section 5.6) in particular
may not transfer.

**Human graders were not blinded** to the study, and grading order was not randomised.

**Model versions are a snapshot.** These are specific model versions accessed over a bounded
period, and provider-side updates can change behaviour without notice. Absolute numbers should
be treated as dated, though the protocol should outlast them.

**Cost was not measured.** We report latency but not API cost per submission, which is likely
to bind at scale and which varies by more than an order of magnitude across the models tested.

---

## 10. Conclusion

We evaluated seven frontier language models as rubric graders on two engineering assignments,
each graded independently by three humans, with every model run three times, for 945 model
gradings in total.

Methodologically, inter-human agreement has to be measured before any claim about LLM grading
accuracy becomes interpretable, and we demonstrated it rather than asserting it. On the same
Operating Systems data, a conventional single-reference, run-pooled analysis reverses the sign
of measured bias for five of seven models, reports highly significant bias for the two
best-calibrated models, and reports no significant bias for a model that grades significantly
harshly. The two courses' human ceilings differ by a factor of nearly two in ICC, so no
universal bar exists to hold an LLM grader to. That also gives us a candidate explanation for
the opposite verdicts in the published literature: holding models and protocol fixed and
changing only the human panel moves the best model from 2.80x to 1.80x outside human
variability. We suggest the field's disagreement concerns reference quality substantially, and
we note the meta-analytic prediction that would test it.

On the empirical side, no model tested reaches human variability in either course. The best model
disagrees with the human panel 2.80x as much as the humans disagree with each other on
Operating Systems, and 1.76x on Biomaterials, and inserting any model as an additional rater
lowers panel reliability in both. That holds despite Pearson correlations reaching 0.88, which
we read as evidence that correlation alone is an inadequate reporting standard for grading
applications.

For practice, what matters most is that the dominant failure is a systematic harshness
offset and not random error, with twelve of fourteen model-course pairs significantly harsh,
and per-rubric calibration can correct that in a way it could not correct random error. This
supports assistive deployments, including triage, second-opinion flagging, and calibrated
first-pass scoring under human review. It does not support autonomous grading of record.

We recommend that studies of LLM grading report three things as a minimum. Inter-human
reliability for the same items, which is the ceiling. Absolute error expressed relative to
human-human disagreement, which is the honest accuracy measure. Run-to-run variability across
repeated invocations, which single-run studies cannot see and which determines whether
identical work receives identical grades.

---

## Reproducibility

All analyses derive from two Jupyter notebooks applying the identical protocol:
`Grading_Dataset_OS/llm_vs_human_grading_analysis.ipynb`, where Phase 9 implements the
human-ceiling protocol, and `Biomaterials/multiLLM_multiHuman_analysis.ipynb`. The grading
pipeline is `grading_service.py`. The experiment drivers are
`Grading_Dataset_OS/test_grading_rerun.py` and `Biomaterials/grade_biomaterials.py`. Figures
3, 9, and 10 come from `docs/paper/make_paper_figures.py`. Grades, both human and model, sit in
`Grading_Dataset_OS/consolidated_results_rerun_4/` and `Biomaterials/grading_results/`.

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
suppl. 1, pp. i21–i29, Jul. 2025. doi: 10.1093/bioinformatics/btaf196

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

[15] J. Zhuo, S. Zhang, X. Fang, H. Duan, D. Lin, and K. Chen, "ProSA: Assessing and
understanding the prompt sensitivity of LLMs," in *Findings of the Association for
Computational Linguistics: EMNLP 2024*, 2024. arXiv:2410.12405

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
*Applied Psychological Measurement*, vol. 24, no. 4, pp. 339–353, 2000.
doi: 10.1177/01466210022031796

---
<!-- 
> **Verification note.** Every reference was confirmed against the source PDF in `Lit/`, an
> author-supplied record, or the publisher/arXiv record. Reference [15] cites the ACL
> Anthology version; its page numbers can be added from that record if the target venue
> requires them for conference papers. -->
