# Statistical Power, Minimum Detectable Effect Size, and Equivalence Testing
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
