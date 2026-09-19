# Dataset -- COMPAS Recidivism (ProPublica)

## The problem

In 2016, ProPublica investigated COMPAS, a risk-assessment algorithm
actually used by courts in Broward County, Florida, to help inform
bail and sentencing decisions. COMPAS scores a defendant's likelihood
of reoffending on a 1-10 scale; judges could see that score when
deciding, among other things, whether someone should be released
before trial. ProPublica obtained COMPAS's scores for thousands of
defendants and matched them against what actually happened over the
following two years, then published the data.

This dataset is that data: each row is one defendant, with their
demographics and criminal history at the time of screening, COMPAS's
own risk score for them, and whether they were actually rearrested
within two years.

**Your task:** predict `two_year_recid` -- will this person be
rearrested within two years? -- from the case facts. Once you have a
model, the more interesting question is the one ProPublica actually
asked: is it equally accurate for everyone, or does it get things
wrong more often, in a particular direction, for some groups than
others? `race` is deliberately excluded from the model's own inputs
(see `config.yaml` and `src/preprocessing.py`) so it can be used
afterward purely to check this, in `src/evaluate.py`.

Before any of that: look at the data first. It comes from a real
system with real data-entry and record-keeping quirks -- don't assume
every column is clean or consistent just because it loads without
error.

## Data dictionary

| column | type | description | notable values |
|--------|------|--------------|------------------|
| `id` | identifier | internal record id | not a model feature |
| `sex` | categorical | defendant's sex | `Male`, `Female` |
| `age` | numeric | defendant's age (years) at screening | |
| `age_cat` | categorical | age bucket | `Less than 25`, `25 - 45`, `Greater than 45` |
| `race` | categorical | defendant's race, as recorded | `African-American`, `Caucasian`, `Hispanic`, `Asian`, `Native American`, `Other`; excluded from model features, used only to audit fairness |
| `juv_fel_count` | numeric | number of prior juvenile felony offenses | |
| `juv_misd_count` | numeric | number of prior juvenile misdemeanor offenses | |
| `juv_other_count` | numeric | number of other prior juvenile offenses | |
| `juvenile_total` | numeric | total juvenile offenses | |
| `priors_count` | numeric | number of prior adult offenses | |
| `prior_offenses` | numeric | number of prior offenses | |
| `age_in_months` | numeric | age expressed in months | |
| `c_charge_degree` | categorical | degree of the current charge | `F` (felony), `M` (misdemeanor) |
| `decile_score` | numeric | COMPAS's own risk score | 1 (lowest risk) to 10 (highest risk); excluded from model features, used only for comparison |
| `score_text` | categorical | COMPAS's own risk category | `Low`, `Medium`, `High`; excluded from model features, used only for comparison |
| `two_year_recid` | binary | **target** -- was this person rearrested within two years? | `0` = no, `1` = yes |

Source: derived from [propublica/compas-analysis](https://github.com/propublica/compas-analysis) (the data behind the "Machine Bias" investigation). Personally-identifying columns (name, date of birth, case numbers, charge descriptions) were removed.


# Week 1


### Hyperparameter Configuration
To address convergence issues and ensure a controlled comparison, the models were trained with the following hyperparameter adjustments:
* **Logistic Regression:** `max_iter = 1000`
* **Decision Tree:** `max_depth = 5`, `min_samples_split = 10`

### Performance and Fairness Comparison
Focusing on the groups with the most statistically significant sample sizes (African-American, n=303; Caucasian, n=232), Table 1 summarizes the core metrics.

| Metric | Logistic Regression | Decision Tree |
| :--- | :---: | :---: |
| Test Accuracy | 67.9% | 66.8% |
| Train-Test Gap | 0.000 (No Overfitting) | +0.012 (Slight Overfitting) |
| FPR (African-American) | 33% | 39% |
| FPR (Caucasian) | 24% | 31% |
| **FPR Difference** | **9 p.p.** | **8 p.p.** |

*Table 1: Performance and False Positive Rate (FPR) comparison.*

1. **Overall Performance:** Both models achieved a similar general accuracy of approximately 67%. The Logistic Regression proved slightly more robust, exhibiting zero train-test gap, whereas the Decision Tree showed a very minor tendency to overfit (+0.012).
2. **Algorithmic Fairness:** Both models exhibited racial bias with higher False Positive Rates for African-American defendants (33% in LogReg, 39% in Decision Tree) compared to Caucasian defendants (24% in LogReg, 31% in Decision Tree).
3. **Convergence Limitations (Logistic Regression):** Despite increasing the iterations to `max_iter = 1000`, the Logistic Regression model still triggered a convergence warning. This typically indicates that the dataset's features operate on vastly different scales. The optimal solution is not to increase iterations further, but to implement feature scaling in the preprocessing pipeline.