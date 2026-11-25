# Principles of Inferential Statistics, Experimental Design, and Hypothesis Testing
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
