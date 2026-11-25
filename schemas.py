"""
Agent-specific data models — fully decoupled from the backend DB models.

These Pydantic models are the *lingua franca* between Tools, Skills,
and the Agent orchestrator.  The backend maps its SQLModel objects to/from
these when calling the agent.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Research context  (input to the agent)
# ---------------------------------------------------------------------------
class ResearchContext(BaseModel):
    """Everything the agent needs to know about a research instance."""

    instance_id: Optional[UUID] = None
    name: Optional[str] = None
    problem_statement: Optional[str] = None
    hypothesis: Optional[str] = None
    research_type: Optional[str] = None

    # Variables
    independent_variable: Optional[str] = None
    dependent_variable: Optional[str] = None
    independent_variable_type: Optional[str] = None
    dependent_variable_type: Optional[str] = None
    covariates: list[str] = Field(default_factory=list)
    time_variable: Optional[str] = None
    subject_group: Optional[str] = None

    # Dataset metadata
    file_path: Optional[str] = None
    processed_file_path: Optional[str] = None
    sample_size: Optional[int] = None
    dataset_preview: Optional[str] = None
    dataset_columns: list[str] = Field(default_factory=list)

    # Participant info
    subjects_or_participants: Optional[str] = None
    inclusion_criteria: Optional[str] = None
    data_collection_period: Optional[str] = None
    collection_location: Optional[str] = None

    # Summaries (computed)
    independent_variable_summary: Optional[str] = None
    dependent_variable_summary: Optional[str] = None
    dataset_summary: Optional[str] = None

    # Domain & Repository Metadata (generic, configured per project/dataset)
    dataset_source_name: Optional[str] = None
    dataset_version: Optional[str] = None
    variable_units: dict[str, str] = Field(default_factory=dict)
    factor_level_definitions: dict[str, str] = Field(default_factory=dict)
    unmeasured_confounders_note: Optional[str] = None
    sampling_boundaries_note: Optional[str] = None
    analysis_artifact_dir: Optional[str] = None
    data_repository_url: Optional[str] = None
    code_repository_url: Optional[str] = None
    provenance_notes: Optional[str] = None
    foundational_literature: list[LiteratureItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Variable info
# ---------------------------------------------------------------------------
class VariableInfo(BaseModel):
    name: str
    detected_type: Optional[str] = None   # numeric | categorical | binary | None
    summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Structural metadata prediction result
# ---------------------------------------------------------------------------
class StructuralMetadata(BaseModel):
    independent_variable: Optional[str] = None
    dependent_variable: Optional[str] = None
    research_type: Optional[str] = None
    time_variable: Optional[str] = None
    subject_group: Optional[str] = None


# ---------------------------------------------------------------------------
# Statistical test result
# ---------------------------------------------------------------------------
class PairwiseContrast(BaseModel):
    group1: str
    group2: str
    mean1: float
    mean2: float
    diff: float
    p_value: float
    significant: bool
    direction: str  # e.g., "Group A (mean=10.50) is higher than Group B (mean=8.20)"
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class RegressionCoefficient(BaseModel):
    parameter: str
    coef: float
    std_err: float
    t_value: float
    p_value: float
    ci_lower: float
    ci_upper: float

    @property
    def variable(self) -> str:
        return self.parameter

    @property
    def coefficient(self) -> float:
        return self.coef


# ---------------------------------------------------------------------------
# Data Filtering & Provenance Manifest (Domain-Agnostic Single Source of Truth)
# ---------------------------------------------------------------------------
class DataFilteringManifest(BaseModel):
    """
    Immutable data audit trail generated dynamically from code execution.
    Tracks raw sample sizes, exclusions, outlier fences, transformations, and filters.
    Contains zero study-specific defaults.
    """
    raw_sample_size: int = Field(..., description="Initial raw row count before processing")
    final_sample_size: int = Field(..., description="Cleaned row count used in inferential modeling")
    outliers_detected: int = Field(default=0, description="Count of candidate outliers identified")
    outliers_removed: int = Field(default=0, description="Count of outlier records excluded from analysis")
    missing_rows_removed: int = 0
    outlier_method: Optional[str] = Field(default=None, description="Detection algorithm (e.g. IQR 1.5, Z-score > 3.0)")
    outlier_variable: Optional[str] = Field(default=None, description="Column name screened for outliers")
    q1: Optional[float] = Field(default=None, description="First quartile (25th percentile)")
    q3: Optional[float] = Field(default=None, description="Third quartile (75th percentile)")
    iqr: Optional[float] = Field(default=None, description="Interquartile range (q3 - q1)")
    lower_fence: Optional[float] = Field(default=None, description="Exact lower outlier fence (q1 - 1.5*iqr)")
    upper_fence: Optional[float] = Field(default=None, description="Exact upper outlier fence (q3 + 1.5*iqr)")
    variable_transformations: dict[str, str] = Field(default_factory=dict, description="Variables scaled or transformed")
    filters_applied: list[dict[str, Any]] = Field(default_factory=list, description="Audit of row filters applied")


class StudyFacts(BaseModel):
    """
    Immutable single source of truth for the entire research paper pipeline.
    Extracted deterministically from the dataset and statistical test outputs.
    Guarantees that Abstract, Intro, Methods, Results, Discussion, and Conclusion
    never describe conflicting sample sizes, degrees of freedom, or directional effects.
    """

    sample_size_total: int
    geographic_scope: Optional[str] = "Multi-site empirical dataset"
    provenance_repository: Optional[str] = "Primary data repository"
    material_analyzed: Optional[str] = "Empirical observational units"
    measurement_scale: Optional[str] = None
    secondary_scale: Optional[str] = None
    isotopic_scale: Optional[str] = None
    water_scale: Optional[str] = None
    manifest: Optional[DataFilteringManifest] = None

    factor_name: str
    factor_levels: list[str]
    reference_level: str
    level_sample_sizes: dict[str, int]
    level_descriptives: dict[str, dict[str, float]] = Field(default_factory=dict)

    dependent_variable: str
    covariates: list[str] = Field(default_factory=list)
    covariates_scaling: dict[str, str] = Field(default_factory=dict)

    anova_f: float
    anova_df_between: int
    anova_df_within: int
    anova_p: float
    eta_squared: float
    eta_squared_ci: tuple[float, float] = (0.0, 0.0)

    kruskal_h: float
    kruskal_df: int
    kruskal_p: float

    regression_model_type: str = "Ordinary Least Squares (OLS) with Dummy Contrasts"
    regression_r_squared: float = 0.0
    regression_coefficients: list[RegressionCoefficient] = Field(default_factory=list)

    lmm_executed: bool = False
    lmm_icc: Optional[float] = None
    lmm_variance_site: Optional[float] = None
    lmm_variance_residual: Optional[float] = None
    cluster_variable: Optional[str] = None
    num_clusters_total: Optional[int] = None
    level_cluster_counts: dict[str, int] = Field(default_factory=dict)
    effective_sample_size: Optional[float] = None
    design_effect: Optional[float] = None
    lmm_fixed_effects: list[RegressionCoefficient] = Field(default_factory=list)
    lmm_marginal_r2: Optional[float] = None
    lmm_conditional_r2: Optional[float] = None

    pairwise_contrasts: list[PairwiseContrast] = Field(default_factory=list)

    # Pre-screening raw cohort sensitivity check metrics
    raw_sample_size: Optional[int] = None
    raw_anova_f: Optional[float] = None
    raw_anova_df_between: Optional[int] = None
    raw_anova_df_within: Optional[int] = None
    raw_anova_p: Optional[float] = None
    raw_eta_squared: Optional[float] = None

    # Covariate descriptive statistics (ranges, SDs, medians)
    covariate_descriptives: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Robust regression standard errors (Huber-White HC3)
    ols_robust_se: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Model diagnostics statistics
    shapiro_wilk_stat: Optional[float] = None
    shapiro_wilk_p: Optional[float] = None
    breusch_pagan_stat: Optional[float] = None
    breusch_pagan_p: Optional[float] = None
    vif_max: Optional[float] = None

    # Power analysis and Minimum Detectable Effect Sizes
    mdes_by_factor_level: dict[str, float] = Field(default_factory=dict)

    # Variance decomposition shares and methodology
    cluster_robust_ols: dict[str, dict[str, float]] = Field(default_factory=dict)
    additional_diagnostics: dict[str, Any] = Field(default_factory=dict)
    variance_shares: dict[str, float] = Field(default_factory=dict)
    variance_share_method: str = "Linear predictor variance decomposition Var(beta_k * X_k) / Var(X*beta) (Nakagawa & Schielzeth, 2013)"

    # Substantive provenance and measurement metadata
    dataset_source_name: Optional[str] = None
    dataset_version: Optional[str] = None
    variable_units: dict[str, str] = Field(default_factory=dict)
    factor_level_definitions: dict[str, str] = Field(default_factory=dict)
    unmeasured_confounders_note: Optional[str] = None
    sampling_boundaries_note: Optional[str] = None
    mineralogy_note: Optional[str] = None
    age_window_note: Optional[str] = None

    @property
    def icc(self) -> Optional[float]:
        """Convenience alias for lmm_icc."""
        return self.lmm_icc

    @property
    def marginal_r2(self) -> Optional[float]:
        return self.lmm_marginal_r2

    @property
    def conditional_r2(self) -> Optional[float]:
        return self.lmm_conditional_r2

    @property
    def pairwise_comparisons(self) -> list[PairwiseContrast]:
        """Convenience alias for pairwise_contrasts."""
        return self.pairwise_contrasts

    def to_prompt_context(self) -> str:
        """Render an explicit, non-negotiable prompt context block."""
        coef_lines = []
        for c in self.regression_coefficients:
            coef_lines.append(
                f"  - {c.parameter}: coef = {c.coef:.4f}, SE = {c.std_err:.4f}, "
                f"t = {c.t_value:.2f}, p = {c.p_value:.4f}, 95% CI [{c.ci_lower:.4f}, {c.ci_upper:.4f}]"
            )
        coefs_formatted = "\n".join(coef_lines) if coef_lines else "  - None"

        contrast_lines = []
        for pw in self.pairwise_contrasts:
            sig_marker = "(Statistically Significant)" if pw.significant else "(Not Significant)"
            ci_str = f", 95% CI [{pw.ci_lower:.2f}, {pw.ci_upper:.2f}]" if pw.ci_lower is not None and pw.ci_upper is not None else ""
            contrast_lines.append(f"  - {pw.direction} (diff = {pw.diff:+.3f}, p = {pw.p_value:.4f}{ci_str}) {sig_marker}")
        contrasts_formatted = "\n".join(contrast_lines) if contrast_lines else "  - None"

        desc_lines = []
        for lvl, d in self.level_descriptives.items():
            desc_lines.append(f"  - {lvl} (n={self.level_sample_sizes.get(lvl, 0)}): mean = {d.get('mean', 0.0):.2f}, std = {d.get('std', 0.0):.2f}, median = {d.get('median', 0.0):.2f}")
        desc_formatted = "\n".join(desc_lines) if desc_lines else "  - None"

        lmm_info = "  - No converged mixed-effects estimates available"
        if self.lmm_executed and self.lmm_icc is not None:
            cluster_desc = f"{self.num_clusters_total} unique clusters ({self.cluster_variable})" if self.num_clusters_total else "multiple sites"
            eff_desc = f"Effective Sample Size (N_eff) = {self.effective_sample_size:.1f} (Design Effect = {self.design_effect:.2f})" if self.effective_sample_size else ""
            level_cluster_str = ", ".join([f"{k}: {v} sites" for k, v in self.level_cluster_counts.items()]) if self.level_cluster_counts else ""
            
            lmm_fixed_lines = []
            for c in self.lmm_fixed_effects:
                lmm_fixed_lines.append(
                    f"    * LMM {c.parameter}: beta = {c.coef:.4f}, SE = {c.std_err:.4f}, z = {c.t_value:.2f}, p = {c.p_value:.4f}, 95% CI [{c.ci_lower:.4f}, {c.ci_upper:.4f}]"
                )
            fixed_str = "\n".join(lmm_fixed_lines) if lmm_fixed_lines else ""

            lmm_info = (
                f"  * Clustered Data Structure: {cluster_desc}; {eff_desc}\n"
                f"  * Clusters per {self.factor_name}: {level_cluster_str}\n"
                f"  * Intra-Class Correlation (ICC) = {self.lmm_icc:.4f} (~{round(self.lmm_icc * 100, 1)}% between-site variance)\n"
                f"  * Cluster random effect variance = {self.lmm_variance_site}, residual variance = {self.lmm_variance_residual}\n"
                f"  * LMM Fixed Effect Estimates:\n{fixed_str}"
            )

        manifest_info = ""
        if self.manifest:
            m = self.manifest
            manifest_info = (
                f"- Data Provenance & Cleaning Audit:\n"
                f"  * Raw Sample Size: {m.raw_sample_size}\n"
                f"  * Outliers Detected/Removed: {m.outliers_detected} detected, {m.outliers_removed} removed\n"
                f"  * Outlier Method: {m.outlier_method or 'None'} on variable '{m.outlier_variable or self.dependent_variable}'\n"
                f"  * Fences: Q1 = {m.q1}, Q3 = {m.q3}, IQR = {m.iqr}, Lower Fence = {m.lower_fence}, Upper Fence = {m.upper_fence}\n"
                f"  * Variable Transformations: {m.variable_transformations or 'None'}\n"
                f"  * Subsetting Filters Applied: {m.filters_applied or 'None'}\n"
            )

        # Pre-screening raw cohort sensitivity check lines
        sensitivity_str = ""
        if self.raw_sample_size and self.raw_sample_size > self.sample_size_total:
            f_str = f"F({self.raw_anova_df_between}, {self.raw_anova_df_within}) = {self.raw_anova_f:.4f}, p = {self.raw_anova_p:.5f}, eta^2 = {self.raw_eta_squared:.4f}" if self.raw_anova_f is not None else "N/A"
            sensitivity_str = (
                f"- Pre-Screening Raw Cohort Sensitivity Check (N_raw = {self.raw_sample_size}):\n"
                f"  * Omnibus ANOVA on unfiltered raw cohort: {f_str}\n"
                f"  * Contrast with analytical cohort (N = {self.sample_size_total}): Compare the reported estimates without assuming that cleaning changed significance.\n"
            )

        # Covariate descriptives
        cov_desc_lines = []
        for c_name, cd in self.covariate_descriptives.items():
            cov_desc_lines.append(f"  * {c_name}: mean = {cd.get('mean')}, SD = {cd.get('std')}, range = [{cd.get('min')}, {cd.get('max')}], median = {cd.get('median')}")
        cov_desc_str = "\n".join(cov_desc_lines) if cov_desc_lines else "  - None"

        # Robust regression SE lines
        robust_lines = []
        for p_name, r_info in self.ols_robust_se.items():
            robust_lines.append(f"  * OLS HC3 {p_name}: beta = {r_info.get('coef')}, HC3 SE = {r_info.get('std_err_hc3')}, p = {r_info.get('p_value')}, 95% CI [{r_info.get('ci_lower')}, {r_info.get('ci_upper')}]")
        robust_str = "\n".join(robust_lines) if robust_lines else "  - Not evaluated"

        # MDES lines
        mdes_lines = []
        for lvl, m_val in self.mdes_by_factor_level.items():
            mdes_lines.append(f"  * {lvl}: MDES (80% power) = {m_val:.2f}")
        mdes_str = "\n".join(mdes_lines) if mdes_lines else "  - Not evaluated"

        scale_str = f" ({self.isotopic_scale})" if self.isotopic_scale else ""
        source_str = f" ({self.dataset_source_name})" if self.dataset_source_name else ""
        return (
            "=== IMMUTABLE STUDY FACTS (MANDATORY GROUND TRUTH - DO NOT DEVIATE) ===\n"
            f"- Total Sample Size (N): {self.sample_size_total} complete observations\n"
            f"- Data Repository & Provenance: {self.provenance_repository or 'Primary empirical dataset'}{source_str}\n"
            f"- Geographic Scope: {self.geographic_scope or 'Multi-site empirical dataset'}\n"
            f"- Material / Unit Analyzed: {self.material_analyzed or 'Empirical observations'}\n"
            f"- Dependent Variable: '{self.dependent_variable}'{scale_str}\n"
            f"{manifest_info}"
            f"{sensitivity_str}"
            f"- Independent Variable: '{self.factor_name}' with {len(self.factor_levels)} categories: {', '.join(self.factor_levels)}\n"
            f"  * Baseline Reference Category: '{self.reference_level}'\n"
            f"- Descriptive Statistics by Group:\n{desc_formatted}\n"
            f"- Continuous Covariate Descriptives:\n{cov_desc_str}\n"
            f"- Covariates Controlled: {', '.join(self.covariates) if self.covariates else 'None'}\n"
            f"- Inferential Omnibus Test Results (Analytical Cohort N = {self.sample_size_total}):\n"
            f"  * One-Way ANOVA: F({self.anova_df_between}, {self.anova_df_within}) = {self.anova_f:.4f}, p = {self.anova_p:.5f}, eta^2 = {self.eta_squared:.4f}\n"
            f"  * Kruskal-Wallis Non-Parametric: H({self.kruskal_df}) = {self.kruskal_h:.4f}, p = {self.kruskal_p:.5f}\n"
            f"- Multiple Regression Model ({self.regression_model_type}, R^2 = {self.regression_r_squared:.4f}):\n{coefs_formatted}\n"
            f"- Huber-White HC3 Heteroscedasticity-Robust Standard Errors (OLS):\n{robust_str}\n"
            f"- Linear Mixed-Effects Model (Hierarchical Clustering):\n{lmm_info}\n"
            f"  * Nakagawa Marginal R^2_m (fixed alone): {self.lmm_marginal_r2 if self.lmm_marginal_r2 is not None else 'N/A'}\n"
            f"  * Nakagawa Conditional R^2_c (fixed + random): {self.lmm_conditional_r2 if self.lmm_conditional_r2 is not None else 'N/A'}\n"
            f"  * Fixed-Predictor Variance Share Decomposition: {self.variance_shares if self.variance_shares else 'N/A'} (Method: {self.variance_share_method})\n"
            f"- Minimum Detectable Effect Sizes (MDES at 80% Power, alpha = 0.05):\n{mdes_str}\n"
            f"- Pairwise Contrasts & Empirical Direction of Effect (Tukey HSD - Unadjusted for Clustering):\n{contrasts_formatted}\n"
            f"- Model Diagnostics: Shapiro-Wilk residual normality p = {self.shapiro_wilk_p} (W = {self.shapiro_wilk_stat}), Breusch-Pagan p = {self.breusch_pagan_p} (LM = {self.breusch_pagan_stat}), Max VIF = {self.vif_max}\n"
            "========================================================================\n"
            "CRITICAL EPISTEMIC HONESTY RULES:\n"
            "1. Pseudo-replication & LMM primacy: When clustering is present, report the number of unique clusters per group and intraclass correlation (ICC). If ICC is substantial, hierarchical mixed modeling is the primary inferential standard.\n"
            "2. Non-significant contrasts & wide CIs: When fixed effects fail to reach statistical significance (p > 0.05) and confidence intervals are wide, the data are statistically INCONCLUSIVE. Do not claim evidence of a zero or modest effect—state that the data are inconclusive.\n"
            "3. Directional shifts under adjustment: If covariate adjustment alters the sign or magnitude of regression slopes, report both unadjusted and adjusted values accurately without claiming confirmed confounding when estimates are statistically indistinguishable from zero.\n"
            "4. Variance partitioning: Explicitly report marginal R^2_m, conditional R^2_c, and ANOVA eta^2, distinguishing variance explained by fixed environmental predictors from clustering.\n"
            "5. In Results, NEVER claim 'robust empirical confirmation' when primary model fixed effects are non-significant (p > 0.05).\n"
            "6. Diagnostic reasoning: Heteroscedasticity and non-normality motivate robust standard errors (Huber-White HC3) on OLS, NOT the LMM. The LMM is motivated by cluster non-independence (ICC).\n"
            "7. Uniform contrast polarity: Always maintain uniform sign polarity (Group minus Baseline reference level). NEVER flip polarity between contrasts.\n"
            "8. Banned terms: Never use the word 'dissolve' or 'vanish' for attenuated effect sizes. Write 'become statistically inconclusive'.\n"
            "9. Zero conversational text or placeholder names: Never write 'Here is...', 'Note: ...', 'presented in the study facts', or placeholder citations."
        )

    def to_introduction_context(self) -> str:
        """Render study design and background context strictly for Introduction (no results leakage)."""
        cluster_str = f" across {self.num_clusters_total} clusters ({self.cluster_variable})" if self.num_clusters_total else ""
        level_cluster_str = ", ".join([f"{k}: {v} sites" for k, v in self.level_cluster_counts.items()]) if self.level_cluster_counts else ""
        cluster_line = f"  * Cluster Distribution: {level_cluster_str}\n" if level_cluster_str else ""
        prov_str = f"{self.provenance_repository} ({self.dataset_source_name})" if self.dataset_source_name else (self.provenance_repository or 'Primary research database')
        units_str = ", ".join([f"{k}: {v}" for k, v in self.variable_units.items()]) if self.variable_units else "Standard empirical units"
        levels_def_str = "; ".join([f"{k}: {v}" for k, v in self.factor_level_definitions.items()]) if self.factor_level_definitions else ", ".join(self.factor_levels)
        return (
            "=== STUDY DESIGN & SCOPE (FOR INTRODUCTION - NO RESULTS LEAKAGE) ===\n"
            f"- Total Analytical Sample Scope: {self.sample_size_total} observations{cluster_str}\n"
            f"- Data Repository & Provenance: {prov_str}\n"
            f"- Geographic / Setting Scope: {self.geographic_scope or 'Empirical multi-site cohort'}\n"
            f"- Material / Unit Analyzed: {self.material_analyzed or 'Empirical observational units'}\n"
            f"- Measurement Units & Scales: {units_str}\n"
            f"- Independent Variable: '{self.factor_name}' with {len(self.factor_levels)} classifications ({levels_def_str})\n"
            f"  * Baseline Reference Category: '{self.reference_level}'\n"
            f"{cluster_line}"
            f"- Controlled Covariates: {', '.join(self.covariates) if self.covariates else 'None'}\n"
            f"- Methodological Framework: Evaluates differences using parametric and non-parametric tests, multivariate regression with robust errors, and hierarchical mixed-effects models.\n"
            "====================================================================\n"
            "CRITICAL INTRODUCTION WRITING RULES:\n"
            "1. ZERO RESULTS LEAKAGE: Do NOT report empirical test findings (eta^2, F-statistics, p-values, or ICC) in the Introduction. The Introduction sets the stage before results are known.\n"
            "2. VERIFIED CITATIONS ONLY: Cite ONLY papers present in the verified literature list. Do NOT invent phantom citations or unverified author names.\n"
            "3. MANDATORY IN-TEXT CITATIONS: Ground claims about prior literature, regional studies, and conflicting findings with in-text citations from the literature list.\n"
            "4. OBJECTIVE HYPOTHESIS: Frame the hypothesis neutrally as estimating the direction, magnitude, and confidence intervals of partial regression offsets under cluster-adjusted modeling.\n"
            "5. ACCURATE PROVENANCE: Base study description strictly on the provided scope and variables; do not invent unperformed screenings or laboratory procedures."
        )


class TestResult(BaseModel):
    test_name: str
    results: dict[str, Any] = Field(default_factory=dict)
    interpretation: Optional[str] = None


# ---------------------------------------------------------------------------
# Literature item
# ---------------------------------------------------------------------------
class LiteratureItem(BaseModel):
    title: str
    citation: str
    review: str
    doi: str = ""
    authors: list[dict[str, str]] = Field(default_factory=list)
    year: str = ""
    journal: str = ""
    work_type: str = ""
    url: str = ""
    metadata_verified: bool = False
    reviewed_source: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)
    relevance_reason: str = ""
    review_scope: str = ""


# ---------------------------------------------------------------------------
# Visualization spec
# ---------------------------------------------------------------------------
class VisualizationSpec(BaseModel):
    title: str
    mark: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color: Optional[str] = None


# ---------------------------------------------------------------------------
# Full agent output
# ---------------------------------------------------------------------------
class AgentOutput(BaseModel):
    """Aggregated output from a full agent run."""

    hypothesis: Optional[str] = None
    structural_metadata: Optional[StructuralMetadata] = None
    test_results: list[TestResult] = Field(default_factory=list)
    test_interpretation: Optional[str] = None
    abstract: Optional[str] = None
    background: Optional[str] = None
    introduction: Optional[str] = None
    methodology: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    statuses: dict[str, str] = Field(default_factory=dict)
    autonomous_analysis: dict[str, Any] = Field(default_factory=dict)
    literature_review: list[LiteratureItem] = Field(default_factory=list)
    visualizations: list[VisualizationSpec] = Field(default_factory=list)
    confounding_variables: list[str] = Field(default_factory=list)
    paper_latex: Optional[str] = None
    paper_bundle: Optional[dict[str, Any]] = None
    latex_paper_path: Optional[str] = None
    latex_pdf_path: Optional[str] = None
    study_facts: Optional[StudyFacts] = None
