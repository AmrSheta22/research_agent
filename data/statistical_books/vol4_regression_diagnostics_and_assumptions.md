# Regression Diagnostics, Assumption Verification, and Model Robustness
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
