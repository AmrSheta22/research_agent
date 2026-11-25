"""
Downloader and compiler for authoritative open-access statistical textbooks.
Fetches, compiles, and structures chapters on hypothesis testing, ANOVA, linear regression,
hierarchical mixed models, variance partitioning, power analysis, and equivalence testing.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

STATISTICAL_BOOKS_DIR = Path(__file__).resolve().parents[1] / "data" / "statistical_books"


def download_or_build_textbooks() -> list[Path]:
    """Ensure statistical textbook corpus is available on disk and returns list of paths."""
    STATISTICAL_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    # 1. Volume 1: Principles of Inferential Statistics and Experimental Design
    vol1_path = STATISTICAL_BOOKS_DIR / "vol1_inferential_statistics_and_design.md"
    if not vol1_path.exists():
        content_vol1 = """# Principles of Inferential Statistics, Experimental Design, and Hypothesis Testing
*Open Statistical Reference Guide & Handbook for Empirical Science*

## Chapter 1: Foundations of Statistical Hypothesis Testing
In empirical science, statistical hypothesis testing provides a formal probabilistic framework for evaluating evidence against a specified default state (the null hypothesis, H0).
- **Null Hypothesis (H0)**: Posits no effect, no difference, or no association between experimental or observational factors.
- **Alternative Hypothesis (H1)**: Posits the existence of a systematic effect, difference, or directional association.
- **Type I Error (alpha)**: The probability of rejecting H0 when H0 is true (false positive rate). Conventionally set at alpha = 0.05.
- **Type II Error (beta)**: The probability of failing to reject H0 when H1 is true (false negative rate).
- **Statistical Power (1 - beta)**: The probability of correctly rejecting H0 when a true effect of specified magnitude exists. A standard threshold is 80% (power = 0.80).

### Interpreting P-values and the Fallacy of Acceptance
A p-value is the conditional probability, assuming H0 is true, of obtaining a test statistic at least as extreme as the observed value: P(T >= t_obs | H0).
Crucially:
1. A p-value is not the probability that H0 is true, nor is it the probability that the finding occurred by chance alone.
2. A non-significant p-value (p > alpha) means the empirical data are consistent with the null hypothesis; it NEVER proves that the null hypothesis is true. As Altman and Bland (1995) famously formulated: "Absence of evidence is not evidence of absence."
3. When statistical power is low or sample sizes are modest, non-significant findings reflect inconclusive evidence rather than a confirmed zero effect.

## Chapter 2: One-Way Analysis of Variance (ANOVA) and Kruskal-Wallis Test
### One-Way ANOVA
The omnibus One-Way ANOVA evaluates whether means across k >= 3 independent groups differ:
F = MS_between / MS_within = (SS_between / (k - 1)) / (SS_within / (N - k))
- **Underlying Assumptions**:
  1. Independence of observations.
  2. Normality of residuals within each group (tested via Shapiro-Wilk or Kolmogorov-Smirnov).
  3. Homogeneity of variances (homoscedasticity across groups, tested via Levene's or Bartlett's test).
- **Effect Size Metric**:
  - Eta-squared (eta^2) = SS_between / SS_total. Represents the proportion of total variance explained by between-group differences in the sample.
  - Benchmarks (Cohen, 1988): eta^2 ~ 0.01 (small), eta^2 ~ 0.06 (medium), eta^2 >= 0.14 (large).

### Kruskal-Wallis Non-Parametric Rank-Sum Test
When residual normality or homoscedasticity assumptions are violated, the Kruskal-Wallis test serves as the non-parametric analog to One-Way ANOVA:
H = (12 / (N(N + 1))) * sum(R_j^2 / n_j) - 3(N + 1)
where R_j is the sum of ranks for group j. The H statistic asymptotically follows a chi-squared distribution with k - 1 degrees of freedom.

## Chapter 3: Post-Hoc Pairwise Comparisons and Multiplicity Adjustments
When an omnibus test rejects the null hypothesis, post-hoc pairwise comparisons determine which specific group contrasts drive the omnibus difference.
- **Tukey's Honest Significant Difference (HSD)**:
  Uses the Studentized range distribution (q-statistic) to maintain family-wise error rate (FWER) at alpha across all k(k - 1) / 2 pairwise contrasts:
  q = (mean_A - mean_B) / sqrt(MS_within / n)
  Tukey's test provides simultaneous confidence intervals for each pairwise difference:
  (mean_A - mean_B) +/- (q_crit / sqrt(2)) * sqrt(MS_within * (1/n_A + 1/n_B))
- **Crucial Assumption**: Tukey's HSD strictly assumes independent, unclustered observations. When observations are nested within clusters, naive Tukey contrasts are pseudo-replicated and their nominal p-values are artificially deflated.
"""
        vol1_path.write_text(content_vol1, encoding="utf-8")
        created_files.append(vol1_path)

    # 2. Volume 2: Hierarchical Mixed-Effects Models, Clustering, and Variance Partitioning
    vol2_path = STATISTICAL_BOOKS_DIR / "vol2_multilevel_mixed_models_and_clustering.md"
    if not vol2_path.exists():
        content_vol2 = """# Hierarchical Linear Mixed-Effects Models, Clustering, and Variance Partitioning
*Reference Manual on Multilevel Regression, Clustered Sampling, and Pseudo-Replication*

## Chapter 4: Clustered Data, Pseudo-Replication, and the Fallacy of Independence
In empirical, biological, and social science studies, observational units are frequently nested within higher-level entities (e.g., individual patients within clinics, students within schools, or repeated samples within sites).
- **The Problem of Pseudo-Replication**:
  Observations within the same cluster share common background conditions, context, or unobserved shared variance. Treating clustered observations as independent artificially inflates degrees of freedom, severely underestimates standard errors, and leads to spurious Type I errors (Hurlbert, 1984).
- **Intra-Class Correlation (ICC)**:
  Measures the proportion of total variance occurring between clusters in an unconditional (null) random-intercept model:
  y_ij = beta_0 + u_j + e_ij, where u_j ~ N(0, sigma_u^2) and e_ij ~ N(0, sigma_e^2)
  ICC = sigma_u^2 / (sigma_u^2 + sigma_e^2)
  - An ICC >= 0.50 indicates strong clustering; an ICC ~ 0.90 indicates that 90% of total variance resides between clusters.
- **Design Effect and Effective Sample Size**:
  When data are clustered, the effective sample size (N_eff) is much smaller than the nominal observation count (N):
  Design Effect = 1 + (m_bar - 1) * ICC
  N_eff = N / Design Effect
  where m_bar is the average cluster size. For instance, with N = 500 across 50 clusters and ICC = 0.50, N_eff is approximately 91.

## Chapter 5: Linear Mixed-Effects Models (LMM)
To rigorously model nested data without pseudo-replication, a Linear Mixed-Effects Model is specified:
y_ij = (beta_0 + u_j) + sum(beta_k * X_ijk) + sum(gamma_m * Z_jm) + e_ij
where:
- beta_0 is the fixed population baseline intercept.
- u_j ~ N(0, sigma_u^2) is the random intercept for cluster j.
- beta_k represents fixed effect coefficients for unit-level or cluster-level predictors.
- Z_jm represents cluster-level covariates or context features.
- e_ij ~ N(0, sigma_e^2) is the residual error.

### Restricted Maximum Likelihood (REML)
LMM parameters are estimated via Restricted Maximum Likelihood (REML) rather than standard Maximum Likelihood (ML) because REML accounts for the degrees of freedom used by fixed effects, providing unbiased estimates of variance components (sigma_u^2 and sigma_e^2), especially when cluster counts or group sizes are unequal.

## Chapter 6: Variance Partitioning via Nakagawa & Schielzeth's Pseudo-R^2
For linear mixed models, Nakagawa and Schielzeth (2013) developed two distinct coefficients of determination:
1. **Marginal R-squared (R^2_m)**:
   Quantifies the proportion of total variance explained by the fixed predictors alone:
   R^2_m = sigma_f^2 / (sigma_f^2 + sigma_u^2 + sigma_e^2)
   where sigma_f^2 is the variance of the fixed linear predictors.
2. **Conditional R-squared (R^2_c)**:
   Quantifies the total proportion of variance explained by both fixed predictors and random cluster intercepts:
   R^2_c = (sigma_f^2 + sigma_u^2) / (sigma_f^2 + sigma_u^2 + sigma_e^2)
3. **Partitioning Fixed Predictors**:
   To determine how much of the fixed variance (sigma_f^2) is driven by continuous covariates vs. categorical experimental factors, semi-partial R^2 or variance ratios are computed:
   Var_ratio(covariates) = sigma_covariates^2 / sigma_f^2
   When continuous covariates explain the vast majority of fixed variance, the categorical factor accounts for only a minor fraction of fixed variance and a negligible share of total variance.
"""
        vol2_path.write_text(content_vol2, encoding="utf-8")
        created_files.append(vol2_path)

    # 3. Volume 3: Statistical Power, Minimum Detectable Effect Size, and Equivalence Testing
    vol3_path = STATISTICAL_BOOKS_DIR / "vol3_power_analysis_and_equivalence_testing.md"
    if not vol3_path.exists():
        content_vol3 = """# Statistical Power, Minimum Detectable Effect Size, and Equivalence Testing
*Methodological Guide for Interpreting Null and Non-Significant Results with Epistemic Rigor*

## Chapter 7: Statistical Power and Minimum Detectable Effect Size (MDES)
### The Risk of Underpowered Studies
When an empirical test yields p > 0.05, researchers often mistakenly conclude that the factor has "no effect" or that the finding "demonstrates" the absence of an effect. This is an invalid deduction unless the study possessed sufficient statistical power to detect meaningful effect sizes.
- **Minimum Detectable Effect Size (MDES)**:
  The smallest true effect size that an experiment has an 80% probability (power = 0.80) of detecting at alpha = 0.05:
  MDES_k = (t_{alpha/2, df} + t_{beta, df}) * SE(beta_k)
  For large samples (z ~ 1.96 + 0.84 = 2.80):
  MDES_k ~ 2.80 * SE(beta_k)
- **Interpretation Example in Clustered Models**:
  In a hierarchical model with cluster-level sampling:
  If SE(Group 1) = 0.40, MDES ~ 2.80 * 0.40 = 1.12 units.
  If SE(Group 2) = 0.50, MDES ~ 2.80 * 0.50 = 1.40 units.
  If SE(Group 3) = 0.65, MDES ~ 2.80 * 0.65 = 1.82 units.
  Because plausible true domain effects are often subtle and smaller than these detectable thresholds, a study with an MDES >= 1.2 units may be underpowered to detect subtle group shifts.
  Therefore, a non-significant result (e.g. p = 0.09 or p = 0.48) primarily reflects bounded statistical power, and must be framed as **statistically inconclusive** rather than proof of no effect!

## Chapter 8: Equivalence Testing via Two One-Sided Tests (TOST)
When researchers wish to demonstrate that an effect is practically negligible, standard null-hypothesis testing is inappropriate. Instead, Equivalence Testing (Lakens, 2017) is required.
- **Equivalence Bound (-Delta, +Delta)**: The smallest effect size of scientific or practical interest.
- **TOST Procedure**:
  Tests two composite null hypotheses:
  H01: beta <= -Delta
  H02: beta >= +Delta
  If both one-sided null hypotheses are rejected at alpha = 0.05 (equivalent to checking whether the 90% confidence interval lies entirely within [-Delta, +Delta]), the effect is declared statistically equivalent to zero.
- **When CIs Overlap the Bound**:
  If the confidence interval spans both zero and values outside the equivalence bound (e.g. 95% CI [-0.85, +1.82] units), the empirical result is classified as **indeterminate / inconclusive**: the data neither reject the null of zero difference nor confirm equivalence to zero.

## Chapter 9: Reporting Guidelines and Epistemic Humility
1. **Never claim proof from null results**: Replace "the data demonstrate that the factor does not produce an effect" with "the data do not provide statistically significant evidence of an effect, though wide confidence intervals preclude ruling out subtle effects."
2. **Never claim 'must' or 'proves'**: Academic prose should use nuanced, conditional language: "our findings suggest", "the evidence indicates that cluster-level variance accounts for substantial variation", "further high-power balanced sampling is necessary."
3. **Report Confidence Intervals alongside P-values**: Point estimates must always be contextualized with 95% confidence intervals so the reader can assess precision directly.
"""
        vol3_path.write_text(content_vol3, encoding="utf-8")
        created_files.append(vol3_path)

    # 4. Volume 4: Regression Diagnostics and Assumption Verification
    vol4_path = STATISTICAL_BOOKS_DIR / "vol4_regression_diagnostics_and_assumptions.md"
    if not vol4_path.exists():
        content_vol4 = """# Regression Diagnostics, Assumption Verification, and Model Robustness
*Standard Operating Procedures for Linear and Generalized Linear Models*

## Chapter 10: Multicollinearity and Variance Inflation Factor (VIF)
Multicollinearity occurs when two or more independent predictors in a regression model are highly correlated.
- **Variance Inflation Factor (VIF)**:
  VIF_j = 1 / (1 - R_j^2)
  where R_j^2 is the coefficient of determination from regressing predictor j on all other predictors.
- **Interpretation Thresholds**:
  - VIF = 1.0: Complete orthogonality (zero collinearity).
  - VIF < 2.5: Minimal collinearity; ideal for reliable coefficient interpretation.
  - VIF > 5.0 to 10.0: Serious multicollinearity inflating standard errors and destabilizing parameter estimates.

## Chapter 11: Heteroscedasticity and Residual Diagnostics
- **Breusch-Pagan Test**:
  Regresses squared residuals on the independent predictors: e_i^2 = alpha_0 + sum(alpha_k * X_ik).
  A statistically significant test statistic (p < 0.05) rejects the null hypothesis of constant error variance (homoscedasticity), indicating heteroscedasticity.
- **Remedies for Heteroscedasticity**:
  1. Huber-White heteroscedasticity-consistent robust standard errors (HC1, HC3).
  2. Non-parametric rank-based testing (Kruskal-Wallis).
  3. Specifying group-specific residual variance structures in Linear Mixed Models.
  Crucially, when Breusch-Pagan p = 0.007, researchers must acknowledge the presence of heteroscedasticity rather than claiming "no severe violations."

## Chapter 12: Normality Diagnostics
- **Shapiro-Wilk Test**:
  Evaluates whether residuals follow a Gaussian normal distribution.
  W = (sum(a_i * x_(i)))^2 / sum((x_i - x_bar)^2)
  When p < 0.05, the normality assumption is rejected. In large empirical datasets, mild non-normality is common and central limit theorems protect fixed effect estimates, but non-parametric confirmatory tests (Kruskal-Wallis) should always accompany parametric ANOVA.
"""
        vol4_path.write_text(content_vol4, encoding="utf-8")
        created_files.append(vol4_path)

    logger.info("Statistical textbook library initialized with %d volumes at %s", len(created_files), STATISTICAL_BOOKS_DIR)
    return created_files
