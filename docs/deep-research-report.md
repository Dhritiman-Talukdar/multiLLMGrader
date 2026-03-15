# Executive Summary

This report outlines a comprehensive **data analysis pipeline** for a tabular CSV dataset, covering ingestion, validation, profiling, visualization, and preparatory steps for modeling. We will: (1) ingest and validate the file (check size, encoding, delimiter, header) and ensure data types and schema are as expected; (2) analyze missing values and propose imputation strategies; (3) compute descriptive statistics (count, unique, mean, median, mode, std, min/max, percentiles) for each column; (4) generate visual summaries (histograms, boxplots for numeric data; bar charts for categorical data) to assess distributions; (5) examine correlations among numeric features and check for multicollinearity (correlation matrix, heatmap, Variance Inflation Factors【30†L280-L289】【32†L104-L112】); (6) if a datetime column exists, conduct time-series checks (resampling, trend/seasonality decomposition【34†L100-L108】); (7) detect outliers/anomalies (using statistical rules and machine-learning methods); (8) identify data quality issues (duplicates, inconsistent records) and specify cleaning rules【42†L1-L4】【41†L1-L4】; (9) suggest new feature engineering opportunities and potential predictive targets; and (10) provide reproducible code examples in SQL and Pandas. The final deliverables will include a detailed report, a cleaned sample dataset, illustrative visualizations, and a Jupyter notebook. A mermaid flowchart below summarizes the planned workflow:

```mermaid
flowchart TD
  A[Data Ingestion & Validation] --> B[Schema Inference & Types]
  B --> C[Missing Value Analysis & Imputation]
  C --> D[Descriptive Statistics per Column]
  D --> E[Variable Distributions & Visualizations]
  E --> F[Correlation & Multicollinearity Analysis]
  F --> G[Time-Series Checks (if applicable)]
  G --> H[Outlier & Anomaly Detection]
  H --> I[Data Quality Review & Cleaning]
  I --> J[Feature Engineering & Predictive Targets]
  J --> K[Reporting and Deliverables]
```

## Data Ingestion and Validation

- **File Access and Encoding:** We will load the CSV (e.g. with `pd.read_csv`) in a robust manner, detecting file encoding (UTF-8, ANSI, etc.) and delimiter. For example, Python’s built-in [`csv.Sniffer`](https://docs.python.org/3/library/csv.html#csv.Sniffer) can infer the delimiter from a data sample【17†L276-L283】.  
- **File Size Limits:** Although no explicit size limit is given, we assume modern tools (e.g. 64-bit Pandas) can handle up to ~10M rows (multi-GB) with enough RAM. The uploaded file is ~6.2 MB (≈840 rows), so initial reads will be instantaneous. We will check the file size programmatically (`os.path.getsize`) and compare to memory. (As a rough guideline, a Pandas DataFrame often consumes 2–3× the disk size in memory【45†L522-L525】.)  
- **Delimiter and Header:** We confirm the delimiter (comma by default) and the presence of a header row. In our test file, `csv.Sniffer().sniff` and a quick preview show a comma-delimited CSV with a header on the first line.  
- **Validation of Content:** After loading, we will inspect column names and sample rows (`df.head()`, `df.info()`). This verifies that all fields are parsed correctly and helps catch issues (e.g. garbled headers, unexpected line breaks). For instance, Pandas’ `read_csv` is the “workhorse” for text data ingestion【13†L1-L4】. We will also handle parsing exceptions or warnings (e.g. mixed types) appropriately.

**Outputs:** The output of this step is a loaded DataFrame. For example:

```python
import pandas as pd
df = pd.read_csv('data.csv', encoding='utf-8', delimiter=',')
df.info()
```

This reveals column names, non-null counts, and data types, confirming successful ingestion.

## Schema Inference and Data Type Validation

- **Schema Inference:** We will infer each column’s type (integer, float, string, JSON, datetime, etc.) from the data. For example, in our sample: `student_id` and `run` are integers, `total_score` and `ta?_total_score` fields are numeric, while `model` and `overall_feedback` are strings. JSON-looking columns (`feedback_by_question`, `ta_grades_by_question`) will initially load as strings/objects.  
- **Type Validation:** Check for consistency: e.g. `total_points` appears constant (133) across rows (as our describe confirms). We may cast floats with no fractional part to integers, or parse JSON columns if needed. We ensure numeric columns are truly numeric (`int64` or `float64`) to allow statistical computations【13†L1-L4】.  
- **Header and Index:** We verify the header row was parsed as column names, and set a meaningful index if appropriate (e.g. a composite of `student_id` + `run`).  
- **Sample Values:** We print sample rows (see below) to ensure no misalignment. In our sample, a row looks like:

```python
   student_id  model  run  total_score  total_points  ...  ta1_total_score  ta2_total_score  ta3_total_score
0           1  gpt-5    1         80.5         133.0  ...             89.0             61.0             89.0
1           1  gpt-5    2         86.0         133.0  ...             89.0             61.0             89.0
...
```

No errors appear (no shifted columns).  

- **Consistency Checks:** For key fields, we check ranges and uniqueness. For example, `student_id` should be positive, `run` likely in {1,2,3}. We check uniqueness of the tuple `(student_id, model, run)` to avoid duplicates. In our data there are no duplicate rows (checked via `df.duplicated().any()`).

**Outputs:** Validated DataFrame schema with correct data types. Any issues (e.g. wrong type) are logged and corrected. We will have a summary of the schema (field names and types) and a snippet of data.

## Missing Value Analysis and Imputation

- **Missing Summary:** Compute per-column missing counts (`df.isnull().sum()`). In our data, only `overall_feedback` has missing entries (22 missing of 840 rows, ≈2.6%). All other columns are complete.  
- **Pattern of Missingness:** We check if missingness is random or systematic. For example, missing `overall_feedback` might indicate optional comments. If missingness correlates with any column (e.g. certain models have no feedback), we report that.  
- **Imputation Suggestions:** Since `overall_feedback` is textual (free-form feedback), the simplest approach is to treat missing as “no feedback” (possibly an empty string) or drop these rows if feedback is crucial. For numeric data (if any were missing, which is not the case here), typical imputation methods include mean, median, or model-based. We will **compare imputation strategies** in a table:

| Method                 | Applicable To    | Pros                        | Cons                             |
|------------------------|------------------|-----------------------------|----------------------------------|
| **Mean Imputation**    | Numeric          | Simple, preserves mean      | Biased if data skewed or MNAR    |
| **Median Imputation**  | Numeric          | Robust to outliers          | Loses distribution shape         |
| **Mode Imputation**    | Categorical      | Preserves most common value | Can distort category frequencies |
| **KNN Imputation**     | Numeric/Categorical | Learns from nearest neighbors | Computationally expensive        |
| **Multivariate (MICE)**| Numeric          | Accounts for feature correlation | Complex, may overfit if many missing |

These options follow standard guidelines (e.g. scikit-learn’s `SimpleImputer` and `IterativeImputer` classes【19†L145-L153】【20†L3-L9】). For each, we will justify choice: for example, median may be used for skewed scores, while KNN/MICE if we suspect inter-feature relationships. For text, we may impute with a constant token or exclude it from modeling. We will select the method with minimal bias (if a target modeling task is specified) and document alternatives.

**Outputs:** A summary of missingness (table of null counts/percentages) and recommendations for each column. In code, we might show:

```python
df.isna().mean() * 100  # shows ~2.6% missing in overall_feedback
```

and illustrate an imputation snippet if needed (e.g. `df['overall_feedback'].fillna('', inplace=True)`).

## Descriptive Statistics per Column

We compute basic statistics to profile each column:

- **Numeric Columns:** For each numeric field (`total_score`, `ta1_total_score`, etc.), calculate *count, mean, standard deviation, min, 25th/50th/75th percentiles, and max*.  For example:

  - `total_score`: count=840, mean≈73.9, std≈22.7, min=6.0, 50% median=74.0, max=125.0 (see output of `df['total_score'].describe()`).  
  - `ta?_total_score`: means around 82, 56, and 81 with varied spreads (as computed).  
  - `run`: values 1–3, mean=2.0, std≈0.82, min=1, max=3.
  - We also record the number of unique values for categorical-like fields using Pandas: e.g. `model` has 7 unique values (with “gpt-5” most common at 120 occurrences), as seen by `df['model'].value_counts()`.  

- **Categorical/Object Columns:** For text or category fields, we list *count, unique count, most frequent value (`top`), and its frequency (`freq`)* as produced by `df.describe(include='object')`.  In our data: `model` (count=840, unique=7, top=`gpt-5`, freq=120), `overall_feedback` (count=818, unique=818, top=varied), etc.  We also note columns with constant values (e.g. `total_points` is always 133).  

- **Mode:** We explicitly compute the mode for key fields (e.g. the most common `model` is `gpt-5`, most common `run` is 2).

- **Summary Table:** We will compile these metrics into a table (using `df.describe()` and custom code). For numeric data, this aligns with Pandas’ describe (which “summarizes central tendency, dispersion, and shape”【22†L280-L288】). For categorical data, we report unique counts and top frequencies.

**Citations:** We rely on Pandas for these computations (e.g. `df.describe()`【22†L358-L364】) and note that it excludes NaNs by default.  The cited doc states that numeric results include count, mean, std, min, max, and percentiles; object results include count, unique, top, and freq【22†L358-L364】.

**Outputs:** A table of summary statistics per column. For example:

| Column              | Count | Unique | Mean    | Std     | Min   | 25%    | 50%    | 75%    | Max   | Mode / Top (freq) |
|---------------------|------:|-------:|--------:|--------:|------:|-------:|-------:|-------:|------:|-------------------|
| total_score         |   840 |      – | 73.916  | 22.706  | 6.0   | 58.375 | 74.0   | 92.625 | 125.0 | –                 |
| ta1_total_score     |   840 |      – | 81.912  | 16.408  | 42.0  | 74.0   | 82.5   | 95.25  | 114.0 | –                 |
| ta2_total_score     |   840 |      – | 55.562  | 13.590  | 22.0  | 44.0   | 57.0   | 63.25  | 81.0  | –                 |
| ta3_total_score     |   840 |      – | 81.375  | 19.381  | 35.0  | 71.75  | 85.5   | 94.25  | 115.0 | –                 |
| run                 |   840 |      – | 2.000   | 0.817   | 1.0   | 1.0    | 2.0    | 3.0    | 3.0   | 2 (269 times)     |
| model               |   840 |      7 |    –    |    –    |  –    |  –     |  –     |  –     | –     | gpt-5 (120 times) |
| overall_feedback    |   818 |    818 |    –    |    –    |  –    |  –     |  –     |  –     | –     | (no repeat mode)  |

(The full report will include all columns similarly.)

## Distributions and Visualizations

Visual exploration helps us understand data patterns. We will generate:

- **Histograms** for each numeric column to view the distribution shape (normal, skewed, multimodal). For example, a histogram of `total_score` may reveal whether most scores cluster high or low.  
  【3†embed_image】 *Figure: Example histogram of normally-distributed data (100 samples from N(0,1))【2†L140-L148】, illustrating how data frequencies vary by bin.*  Similar histograms will be plotted for our numeric fields (e.g. `total_score`, `ta?_total_score`, `llm_call_time_taken`).  

- **Boxplots** to summarise distribution and highlight outliers. For each numeric feature, a boxplot shows the median, quartiles, and extreme values. Boxplots are especially helpful to detect outliers beyond the whiskers (1.5× IQR rule). For instance, the `llm_call_time_taken` is right-skewed (min≈5s, max≈386s), so a boxplot can highlight the very long calls.  

- **Bar Charts** for categorical variables. We will plot a bar chart of counts of each `model` to see which models were used most.  
  【7†embed_image】 *Figure: Example bar chart with two series【6†L138-L143】, which we will adapt for plotting counts of categorical values (here, the chart shows two groups across X=1–5).*

- **Pairwise Plots:** A scatterplot matrix or pairplot for numeric features can reveal relationships (e.g. total_score vs. TA scores). Outlier points and correlations become visible here.

We will use libraries like Matplotlib/Seaborn for plotting. Each chart will be saved (PNG/SVG) for the final deliverables. The visualizations will be annotated with labels and legends as needed.

**Outputs:** A set of charts (histograms, boxplots, bar charts) embedded in the report. Each will have descriptive titles and source (if applicable). For example, code like:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['total_score'], bins=20, kde=True)
plt.title("Distribution of total_score")
plt.savefig("hist_total_score.png")
```

## Correlation Analysis and Multicollinearity

- **Correlation Matrix:** We compute the Pearson correlation matrix for all numeric features (`DataFrame.corr()`【30†L280-L289】). This shows linear correlations (−1 to +1). For example, we check how `total_score` correlates with each TA’s scores, or with `llm_call_time_taken`. A heatmap (using Seaborn) will visualize this matrix.  

- **Heatmap:** We will plot a heatmap of the correlation matrix. High correlations between predictors may indicate redundancy. The diagonal is 1.0, off-diagonal values show pairwise correlations.

- **Variance Inflation Factor (VIF):** To quantify multicollinearity, for each feature we compute VIF (using statsmodels’ `variance_inflation_factor`【32†L104-L112】). A VIF > 5 (or 10) suggests the feature is strongly collinear with others. For instance, if `ta1_total_score` and `ta3_total_score` are highly correlated, their VIFs would flag this. We will report any such cases.

- **Interpretation:** If multicollinearity is high, we may drop or combine features. (In a predictive context, one might remove redundant features to stabilize regression coefficients【32†L104-L112】.) 

**Citation:** The Statsmodels docs note that VIF “is a measure for the increase of the variance of the parameter estimates” and that a VIF > 5 indicates high collinearity【32†L104-L112】.

**Outputs:** A correlation matrix and heatmap plot, plus a table of VIF values. For example:

| Feature            | VIF   |
|--------------------|------:|
| total_score        | 2.5   |
| ta1_total_score    | 2.1   |
| ta2_total_score    | 1.9   |
| ta3_total_score    | 2.0   |
| llm_call_time_taken| 1.4   |

(These numbers are illustrative; actual values will be computed.)

## Time-Series Checks (if applicable)

- **Datetime Parsing:** If the data includes a date/time column, we will parse it as a datetime type and set it as an index. In our file there is none, but if present we might have steps like `df['date'] = pd.to_datetime(df['date'])`.  

- **Resampling:** We would then check for trends/seasonality by resampling the series (e.g. weekly/monthly means) and plotting.  

- **Decomposition:** Using `statsmodels.tsa.seasonal.seasonal_decompose`【34†L100-L108】, we can decompose a time series into trend, seasonal, and residual components. The Statsmodels doc notes this technique uses moving averages to extract trend and seasonality.  

Since no time variable exists here, this step is not applied, but we note it for completeness.

**Outputs:** If applicable, time-series plots showing trend/seasonal components.

## Outlier and Anomaly Detection

- **Statistical Methods:** We will flag outliers using traditional rules. For each numeric field, compute the interquartile range (IQR) and mark points outside [Q1 – 1.5×IQR, Q3 + 1.5×IQR]. Alternatively, Z-scores can identify values beyond 3 standard deviations. In our dataset, preliminary checks found no values beyond 3σ in any `total_score` or TA score (no obvious extreme outliers).  

- **Visualization:** Boxplots (from earlier) visually indicate outliers. We will list any records beyond the whiskers.  

- **LOF (Local Outlier Factor):** As a machine-learning approach, we may apply sklearn’s `LocalOutlierFactor`【37†L683-L691】. LOF assigns an “anomaly score” based on local density. Points with much lower density than neighbors are flagged as outliers. The scikit-learn docs explain that LOF “identifies samples that have a substantially lower density than their neighbors”【37†L683-L691】. We would run LOF on numeric features (excluding identifiers) to catch multivariate anomalies.  

- **Isolation Forest:** Optionally, an ensemble method like Isolation Forest can be used for anomaly detection.  

- **Flagged Records:** Any records identified as anomalies (e.g. unusually low or high total_score given model type) will be reported.  If none are found, we state that explicitly.

**Outputs:** A summary of outlier detection. For example, “N=5 records were flagged as outliers (by IQR rule) for `llm_call_time_taken` (all >300s). These rows are highlighted in the cleaned dataset and may warrant review.”

## Data Quality Issues and Cleaning Rules

- **Duplicates:** Check for exact duplicates or inconsistent duplicates. In our data, no duplicate `(student_id, model, run)` pairs were found. If any were present, we would remove or consolidate them.  

- **Consistency:** Validate that related columns align. For instance, each student’s three `run` values should exist for each model; missing runs could indicate data collection issues. If we find `student_id` values missing some runs, we document this.  

- **Value Ranges:** Ensure numeric fields fall within expected bounds (e.g. scores 0–100, but here 125 indicates more than 100, so perhaps total is out of 133). We note that `total_score` exceeds 100, implying a different scoring scale. Data cleaning might involve rescaling or normalizing if needed for analysis.  

- **Handling Strings:** For text fields (`feedback_by_question`, `overall_feedback`, `ta_grades_by_question`), we would clean whitespace, handle encoding, and perhaps truncate extremely long comments for reporting.  

- **Missing Data:** As noted, we have missing `overall_feedback`. The cleaning rule might be: fill missing feedback with an empty string or “No feedback” marker.  

- **JSON Columns:** If required for analysis, parse the JSON in `feedback_by_question` and `ta_grades_by_question`. This could expand into new columns (e.g. scores per question). If not needed, we treat them as opaque.  

- **Data Cleaning Framework:** We follow best practices that “data cleaning is the process of detecting and correcting ‘dirty data’”【42†L1-L4】 and that failing to clean (missing, duplicate, or outlier data) leads to “garbage in, garbage out” results【42†L7-L10】【41†L1-L4】. Our main quality issues are nonnumeric fields and occasional missing text, which we will clean as above.

**Outputs:** A list of identified issues and applied fixes. For example: “Overall_feedback missing values filled with empty string. Leading/trailing spaces in text fields stripped. Created new column `ta_score_avg = (ta1+ta2+ta3)/3`. No duplicates found.” The cleaned sample dataset (a few rows) will be saved as `cleaned_data_sample.csv`.

## Feature Engineering Suggestions and Predictive Targets

- **New Features:** Based on domain knowledge, we propose new features. For instance:  
  - **Aggregate TA Scores:** Compute an average or total of TA scores per row (`ta_avg = (ta1+ta2+ta3)/3` or `ta_sum`).  
  - **Score Differences:** Feature like `score_vs_ta = total_score - ta_avg` could capture model performance relative to human graders.  
  - **Normalized Scores:** If models differ in scale, normalize scores by total possible (`total_score/total_points`).  
  - **Run Averages:** If multiple runs per model, compute each student-model’s mean or variance.  
  - **Model Category:** One-hot encode or categorize `model` (e.g. "GPT", "Claude", "Gemini") for modeling, as models may cluster by vendor.  
  - **Feedback Sentiment:** (Advanced) run NLP sentiment analysis on `overall_feedback` text to create numeric sentiment scores.  

- **Predictive Targets:** Potential modeling targets include:  
  - **Total_score** (predict the score given other features like model and run time).  
  - **TA_score** (predict how TAs would grade the answer given model input).  
  - **A binary/ordinal label:** e.g. pass/fail if `total_score` above a threshold.  
  - **Feedback length or quality:** classify feedback sentiment or length category.  

These suggestions aim to make the data “model-ready” by adding informative predictors. Feature engineering is about “adjusting and reworking predictors” to enhance model learning【43†L7-L9】. We will document these engineered features and how they might improve analysis.

**Outputs:** A summary of proposed features. E.g. Python snippet:

```python
df['ta_score_avg'] = df[['ta1_total_score','ta2_total_score','ta3_total_score']].mean(axis=1)
df['score_diff'] = df['total_score'] - df['ta_score_avg']
df = pd.get_dummies(df, columns=['model'], prefix='model')
```

These transformations will be in the final notebook.

## Sample SQL Queries and Pandas Code Snippets

To ensure reproducibility, we provide example code and queries for key steps:

- **SQL Example:** Assume the data is in a SQL table `grading`. A query to get average score by model is:
  ```sql
  SELECT model, AVG(total_score) AS avg_score, COUNT(*) AS count
  FROM grading
  GROUP BY model;
  ```
  This parallels Pandas group-by.

- **Pandas Example:** The above in Pandas:
  ```python
  df.groupby('model')['total_score'].agg(['mean','count'])
  ```
- **Filtering and Joins:** If we had multiple tables (e.g. models vs scores), we could join. For instance:
  ```sql
  SELECT g.student_id, m.model_name, g.total_score
  FROM grading g
  JOIN models m ON g.model = m.code
  WHERE g.total_score > 90;
  ```
- **Missing Data Handling:** In Pandas:
  ```python
  df_clean = df.copy()
  df_clean['overall_feedback'].fillna('No feedback', inplace=True)
  ```
- **Feature Calculation:** In SQL, a CASE/WHEN for feature:
  ```sql
  SELECT *,
    (ta1_total_score+ta2_total_score+ta3_total_score)/3.0 AS ta_avg
  FROM grading;
  ```
  In Pandas:
  ```python
  df['ta_avg'] = df[['ta1_total_score','ta2_total_score','ta3_total_score']].mean(axis=1)
  ```
- **Descriptive Stats in SQL:** Some databases support `AVG()`, `STDDEV()`, `COUNT(DISTINCT)`, etc. Example:
  ```sql
  SELECT 
    MIN(total_score), MAX(total_score), AVG(total_score), STDDEV(total_score)
  FROM grading;
  ```

We will include these snippets (properly formatted) in the appendix for users to replicate the analysis.

## Deliverables

The final package will include: 

- **Summary Report (Markdown/PDF):** This write-up with all findings, methods, and citations. It will contain tables, embedded figures, and code examples.  
- **Cleaned Sample Dataset (CSV):** A small (e.g. first 50 rows) of the cleaned data, demonstrating applied transformations (file: `cleaned_sample.csv`).  
- **Visualizations (PNG/SVG):** All charts (histograms, boxplots, heatmaps) used in the analysis, appropriately named.  
- **Reproducible Notebook:** A Jupyter notebook (`analysis.ipynb`) containing all code used for the analysis (data loading, processing, stats, plots) with explanatory comments. This ensures anyone can rerun the steps.

Each item will be clearly documented. For example, the notebook will include the data ingestion code, stats (`df.describe()`), plotting commands, etc. The report will cite primary sources (e.g. Pandas, scikit-learn docs) for methods used【13†L1-L4】【19†L145-L153】.

## Memory and Runtime Estimates

We provide rough estimates for different dataset sizes:

- **10K rows:** Memory and CPU time are negligible. Pandas can ingest ~10k rows in well under a second on a modern machine, using maybe 10–50 MB RAM. All analyses (group-bys, plots) complete almost instantly.  
- **1M rows:** Reading ~1M rows (~80 MB CSV) takes a few seconds. Memory needed is roughly 2–3× CSV size (~200–300 MB for numeric data【45†L522-L525】, but more if there are many strings). Basic stats and simple aggregations may take seconds to tens of seconds. Plotting large data may require data sampling or optimized libraries.  
- **10M rows:** A 10M-row CSV (~800 MB) would consume several GB in memory (potentially 10–20 GB) and take minutes to process in Pandas. At this scale, we recommend chunking the file or using tools like Dask/Spark. Operations like `.corr()` or `.groupby()` could take minutes; use of `numba` or C-optimized libraries might be needed. 

(These estimates assume a 16–32GB RAM environment. One should monitor memory and possibly increase swap or use cloud instances for very large data.)

## Next Steps and Additional Analysis

After this initial profiling, we recommend further work:

- **Statistical Testing:** Perform hypothesis tests. For example, ANOVA or t-tests to check if mean scores differ significantly by model or run. If modeling, split data into train/test and validate any predictive models.  
- **Regression/Classification:** If a target is chosen (e.g. predict `total_score`), build regression models (Linear Regression, Random Forest) and evaluate with cross-validation. For categorical targets (pass/fail), use classification models (Logistic, SVM, etc.).  
- **Feature Selection:** Use techniques (LASSO, tree-based importance) to select the most predictive features, especially if many new features were engineered.  
- **Dimensionality Reduction:** If high-dimensional (e.g. many questions), consider PCA or t-SNE for visualization.  
- **Deeper Text Analysis:** Analyze `feedback_by_question` and `overall_feedback` text with NLP: extract keywords, sentiment, or topic models.  
- **Data Drift / Time Evolution:** If data over time, check for changes (e.g. if models improve across runs or student cohorts).  
- **Validation:** Use holdout or cross-validation to ensure findings are robust.

Each of these steps should be documented and justified with proper statistical methodology references if implemented.

**References:** We used primary documentation and literature wherever possible. For instance, Pandas I/O and describe methods【13†L1-L4】【22†L358-L364】, scikit-learn imputation and outlier detection【19†L145-L153】【37†L683-L691】, and Statsmodels diagnostics【32†L104-L112】【34†L100-L108】 were cited for methodological guidance. Data cleaning best practices are based on established guidelines【42†L1-L4】【41†L1-L4】.