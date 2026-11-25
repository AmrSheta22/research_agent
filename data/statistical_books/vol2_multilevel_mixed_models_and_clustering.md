# Hierarchical Linear Mixed-Effects Models, Clustering, and Variance Partitioning
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
