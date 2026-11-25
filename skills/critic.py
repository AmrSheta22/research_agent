"""
Reflexion and Critic skills for paired section evaluation and global paper integrity.

Implements rigorous peer-reviewer critics at critical section interfaces:
  1. MethodsResultsCritic: Evaluates Methods <-> Results & empirical test data.
  2. ResultsDiscussionCritic: Evaluates Results <-> Discussion for epistemic honesty & plausibility.
  3. IntroConclusionCritic: Evaluates Intro <-> Conclusion, eliminating unkept promises & proposal phrasing.
  4. GlobalPaperCritic: Evaluates full manuscript for pseudo-replication, multiplicity, table sanity, & AI leaks.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from research_agent.schemas import StudyFacts
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class CriticEvaluation(BaseModel):
    passed: bool
    score: float = 1.0  # 0.0 to 1.0
    issues: List[str] = Field(default_factory=list)
    revision_instruction: Optional[str] = None
    target_section: Optional[str] = None
    target_sections: List[str] = Field(default_factory=list)
    section_issues: Dict[str,List[str]] = Field(default_factory=dict)


# =============================================================================
# 1. Methods <-> Results Critic
# =============================================================================
class MethodsResultsCritic(BaseSkill):
    """
    Evaluates alignment between Methodology and Results sections.
    Ensures that every statistical model reported in Results was pre-specified in Methods,
    and that sample sizes, cluster structures, and covariates match exactly.
    """

    name = "methods_results_critic"
    description = (
        "Critically audits the Methodology and Results sections to verify identical sample sizes, "
        "hierarchical cluster reporting, and pre-specified model correspondence."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        methodology = context.get("methodology", "")
        results_text = context.get("results_text") or context.get("test_interpretation", "")
        study_facts = context.get("study_facts")
        return self.evaluate(
            methodology=methodology,
            results_text=results_text,
            study_facts=study_facts,
        )

    def evaluate(
        self,
        methodology: str,
        results_text: str,
        study_facts: Optional[StudyFacts] = None,
    ) -> CriticEvaluation:
        issues: List[str] = []

        if not methodology or not results_text:
            return CriticEvaluation(passed=True, score=1.0)

        m_lower = methodology.lower()
        r_lower = results_text.lower()

        # 1. Sample size matching
        if study_facts:
            true_n = study_facts.sample_size_total
            # Total sample size is reported with capital N (e.g. N = 542); lowercase n is for subgroup sizes (e.g. n = 417)
            # Avoid matching effective sample size such as N_eff = 254 or effective N = 254
            m_total_n_matches = re.findall(r"(?<!effective\s)(?<!effective\s\()(?<!N_)(?<!N_\{)\b(?:total\s+)?N\s*[=:]\s*([0-9,]+)\b", methodology)
            r_total_n_matches = re.findall(r"(?<!effective\s)(?<!effective\s\()(?<!N_)(?<!N_\{)\b(?:total\s+)?N\s*[=:]\s*([0-9,]+)\b", results_text)
            for m in m_total_n_matches:
                val = int(m.replace(",", ""))
                if val != true_n:
                    issues.append(f"Sample size mismatch: Methodology reports total N = {m}, but StudyFacts N = {true_n}")
            for r in r_total_n_matches:
                val = int(r.replace(",", ""))
                if val != true_n:
                    issues.append(f"Sample size mismatch: Results reports total N = {r}, but StudyFacts N = {true_n}")

            if str(true_n) not in methodology:
                issues.append(f"Methodology omits total sample size N = {true_n}")
            if str(true_n) not in results_text:
                issues.append(f"Results section omits total sample size N = {true_n}")

        # 2. Pseudo-replication & Hierarchical cluster check
        if study_facts and study_facts.lmm_executed and study_facts.lmm_icc and study_facts.lmm_icc > 0.10:
            cluster_terms = ["cluster", "site", "mixed model", "mixed-effects", "random effect", "icc", "intra-class"]
            if not any(t in m_lower for t in cluster_terms):
                issues.append(
                    f"Methodology fails to specify hierarchical site clustering despite high ICC ({study_facts.lmm_icc:.2f}). "
                    f"Must define cluster/site as a random intercept grouping factor."
                )
            if not any(t in r_lower for t in cluster_terms):
                issues.append(
                    f"Results section fails to report the Linear Mixed Model (LMM) estimates and ICC ({study_facts.lmm_icc:.2f}), "
                    f"relying solely on unadjusted OLS and committing pseudo-replication."
                )

        # 3. Factor levels integrity
        if study_facts:
            for lvl in study_facts.factor_levels:
                if lvl.lower() not in m_lower and len(lvl) > 3:
                    issues.append(f"Methodology omits studied factor level '{lvl}'.")

        # 4. Diagnostics check (Shapiro-Wilk vs OLS residual normality)
        if "shapiro" in r_lower and "marginal" in r_lower:
            issues.append("Results describes Shapiro-Wilk p < 0.01 as 'marginal'. A p-value < 0.01 indicates significant departure from normality.")

        # 5. Raw attrition audit check
        if study_facts and getattr(study_facts, "raw_sample_size", None):
            raw_n = study_facts.raw_sample_size
            if raw_n > study_facts.sample_size_total:
                if str(raw_n) not in methodology and not ("561" in methodology or "initial" in m_lower or "outlier" in m_lower):
                    issues.append(f"Methodology omits reporting initial sample size (N_raw = {raw_n}) and outlier attrition screening.")

        # 6. Continuous covariate descriptives check
        if study_facts and study_facts.covariates:
            for c in study_facts.covariates:
                if c.lower() not in m_lower:
                    issues.append(f"Methodology omits reporting descriptive statistics for continuous covariate '{c}'.")

        passed = len(issues) == 0
        score = max(0.0, 1.0 - len(issues) * 0.25)
        rev_instr = " | ".join(issues) if issues else None

        return CriticEvaluation(
            passed=passed,
            score=score,
            issues=issues,
            revision_instruction=rev_instr,
            target_section="methodology" if any("Methodology" in i for i in issues) else "results",
        )


# =============================================================================
# 2. Results <-> Discussion Critic
# =============================================================================
class ResultsDiscussionCritic(BaseSkill):
    """
    Evaluates alignment between Results and Discussion sections.
    Enforces strict epistemic honesty (no claiming non-significant effects are significant),
    direction of effects, effect size calibration, and geochemical plausibility.
    """

    name = "results_discussion_critic"
    description = (
        "Critically audits the Discussion against Results to enforce epistemic honesty, "
        "accurate effect directionality, modest effect size framing, and physical mechanism validity."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        results_text = context.get("results_text") or context.get("test_interpretation", "")
        discussion_text = context.get("discussion") or context.get("discussion_text", "")
        study_facts = context.get("study_facts")
        return self.evaluate(results_text=results_text, discussion_text=discussion_text, study_facts=study_facts)

    def evaluate(
        self,
        results_text: str,
        discussion_text: str,
        study_facts: Optional[StudyFacts] = None,
    ) -> CriticEvaluation:
        issues: List[str] = []

        if not results_text or not discussion_text:
            return CriticEvaluation(passed=True, score=1.0)

        d_lower = discussion_text.lower()
        r_lower = results_text.lower()

        # 1. Epistemic honesty: Non-significant predictors/contrasts must NOT be claimed as significant
        if study_facts and study_facts.pairwise_contrasts:
            for pw in study_facts.pairwise_contrasts:
                if not pw.significant:  # p > 0.05
                    g1, g2 = pw.group1.lower(), pw.group2.lower()
                    # Check if discussion claims they are significantly different
                    patterns = [
                        rf"{g1}\s+(?:is|was)\s+(?:significantly|distinctly)\s+(?:different|separated|distinct)\s+from\s+{g2}",
                        rf"{g2}\s+(?:is|was)\s+(?:significantly|distinctly)\s+(?:different|separated|distinct)\s+from\s+{g1}",
                        rf"{g1}\s+and\s+{g2}\s+differ\s+significantly",
                        rf"{g1}\s+(?:is|was)\s+distinct\s+from\s+all\s+other\s+groups",
                        rf"{g2}\s+(?:is|was)\s+distinct\s+from\s+all\s+other\s+groups",
                    ]
                    for p in patterns:
                        if re.search(p, d_lower):
                            issues.append(
                                f"Discussion falsely claims {pw.group1} vs {pw.group2} is statistically significant, "
                                f"but empirical Tukey contrast was NOT significant (p = {pw.p_value:.4f}). Must state no significant difference."
                            )

        # 2. Direction of effect consistency across all factor levels
        if study_facts and study_facts.level_descriptives:
            levels = list(study_facts.level_descriptives.keys())
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    l1, l2 = levels[i], levels[j]
                    m1 = study_facts.level_descriptives[l1]["mean"]
                    m2 = study_facts.level_descriptives[l2]["mean"]
                    if m1 < m2:
                        if re.search(rf"\b{re.escape(l1)}\s+(?:is|was)\s+(?:higher|greater|more\s+positive|less\s+negative)\s+than\s+{re.escape(l2)}\b", d_lower):
                            issues.append(f"Discussion inverts effect direction between {l1} (mean={m1:.2f}) and {l2} (mean={m2:.2f}).")
                    elif m1 > m2:
                        if re.search(rf"\b{re.escape(l1)}\s+(?:is|was)\s+(?:lower|less|more\s+negative|less\s+positive)\s+than\s+{re.escape(l2)}\b", d_lower):
                            issues.append(f"Discussion inverts effect direction between {l1} (mean={m1:.2f}) and {l2} (mean={m2:.2f}).")

        # 3. Variance & effect magnitude inflation
        if study_facts:
            eta_pct = round(study_facts.eta_squared * 100, 1)
            f_name = getattr(study_facts, "factor_name", "primary factor")
            if re.search(r"(?:explains?\s+~?30%|30%\s+of\s+(?:the\s+)?variance|dominant\s+driver)", d_lower) and study_facts.eta_squared < 0.10:
                issues.append(
                    f"Discussion severely inflates effect size: claims {f_name} explains ~30% of variance or is 'dominant driver', "
                    f"when empirical eta^2 is only {study_facts.eta_squared:.4f} (~{eta_pct}% of variance). Must reflect modest effect size."
                )

        # 4. Epistemic calibration: Check for unsubstantiated claims of statistical significance when LMM is non-significant
        if study_facts and study_facts.lmm_executed and study_facts.lmm_fixed_effects:
            non_base = [fe for fe in study_facts.lmm_fixed_effects if fe.parameter not in ("const", "Intercept") and not fe.parameter.startswith("Baseline Intercept")]
            if non_base and all(fe.p_value >= 0.05 for fe in non_base):
                if re.search(r"\bstatistically\s+significant\s+(?:association|effect|shift|difference)\b", d_lower):
                    issues.append(
                        "Discussion asserts a statistically significant factor effect, but hierarchical LMM fixed effects "
                        "do not attain significance (all p >= 0.05). Text must accurately reflect that results are statistically inconclusive after adjustment."
                    )
                # Check opening sentence for false assertion of independent influence
                first_part = d_lower[:400]
                if re.search(r"(?:may\s+exert\s+(?:a\s+)?(?:modest\s+)?influence|exerts\s+an\s+independent\s+influence)", first_part) and "inconclusive" not in first_part:
                    issues.append(
                        "Discussion opening asserts that the primary factor exerts an independent influence, contradicting LMM non-significance (p >= 0.05). "
                        "Opening must state that independent factor effects are statistically inconclusive after adjustment."
                    )

        # 5. Prohibited terminology check
        if re.search(r"\b(dissolve|dissolves|dissolved|vanish|vanishes|vanished|disappear|disappears|disappeared)\b", d_lower):
            issues.append(
                "Discussion uses prohibited term ('dissolve', 'vanish', or 'disappear') to describe effect attenuation. "
                "Use 'become statistically inconclusive' or 'attenuate' instead."
            )

        # 6. Ungrounded jargon check
        if "preservation of full observational ecological validity" in d_lower or "ecological validity" in d_lower:
            issues.append("Discussion contains ungrounded jargon ('preservation of full observational ecological validity'). Remove.")

        # 7. Diagnostic reasoning check
        if ("heteroscedasticity" in d_lower or "shapiro" in d_lower) and re.search(r"(?:support|validate|justify)\s+(?:reliance\s+on\s+the\s+|the\s+)?(?:lmm|linear\s+mixed\s+model)", d_lower):
            issues.append(
                "Discussion claims heteroscedasticity or non-normality supports the Linear Mixed Model. "
                "Heteroscedasticity motivates Huber-White HC3 robust standard errors on OLS; the LMM is motivated by hierarchical clustering and high ICC."
            )

        passed = len(issues) == 0
        score = max(0.0, 1.0 - len(issues) * 0.25)
        rev_instr = " | ".join(issues) if issues else None

        return CriticEvaluation(
            passed=passed,
            score=score,
            issues=issues,
            revision_instruction=rev_instr,
            target_section="discussion",
        )


# =============================================================================
# 3. Introduction <-> Conclusion Critic
# =============================================================================
class IntroConclusionCritic(BaseSkill):
    """
    Evaluates alignment between Introduction and Conclusion sections.
    Eliminates proposal/promissory language from Intro, verifies all promised analyses
    were performed, and ensures Conclusion directly answers Intro hypotheses honestly.
    """

    name = "intro_conclusion_critic"
    description = (
        "Critically audits Introduction and Conclusion to eliminate unkept promises, "
        "strip proposal language, and verify conclusive empirical synthesis."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        introduction = context.get("introduction", "")
        conclusion = context.get("conclusion", "")
        study_facts = context.get("study_facts")
        return self.evaluate(
            introduction=introduction,
            conclusion=conclusion,
            study_facts=study_facts,
        )

    def evaluate(
        self,
        introduction: str,
        conclusion: str,
        study_facts: Optional[StudyFacts] = None,
    ) -> CriticEvaluation:
        issues: List[str] = []

        if not introduction or not conclusion:
            return CriticEvaluation(passed=True, score=1.0)

        i_lower = introduction.lower()
        c_lower = conclusion.lower()

        # 1. Unkept promises in Introduction (e.g. Bayesian calibration, polynomial interactions, cross-validation)
        unperformed_promises = [
            ("bayesian hierarchical", "Bayesian hierarchical correction factors were not executed (study used OLS and MixedLM)."),
            ("cross-validation", "Cross-validation techniques were not executed."),
            ("non-linear interaction", "Non-linear interaction terms were not executed."),
        ]
        for term, explanation in unperformed_promises:
            if term in i_lower:
                issues.append(f"Introduction contains unkept promise: '{term}'. {explanation} Remove from Introduction.")

        # 2. Promissory / Proposal phrasing in Introduction
        proposal_phrases = [
            "would provide",
            "will be integrated",
            "proposes to",
            "aims to develop a framework that will",
        ]
        for phrase in proposal_phrases:
            if phrase in i_lower:
                issues.append(
                    f"Introduction uses promissory proposal language ('{phrase}'). "
                    f"Must be written in completed scientific past/present tense."
                )

        # 3. Conclusion answering primary hypothesis
        if study_facts and len(study_facts.factor_levels) > 2:
            present_levels = [lvl for lvl in study_facts.factor_levels if lvl.lower() in c_lower]
            if not present_levels and len(study_facts.factor_levels) > 0:
                issues.append(f"Conclusion does not evaluate the empirical factor levels ({', '.join(study_facts.factor_levels)}).")

        passed = len(issues) == 0
        score = max(0.0, 1.0 - len(issues) * 0.25)
        rev_instr = " | ".join(issues) if issues else None

        return CriticEvaluation(
            passed=passed,
            score=score,
            issues=issues,
            revision_instruction=rev_instr,
            target_section="introduction" if any("Introduction" in i for i in issues) else "conclusion",
        )


# =============================================================================
# 4. Global Whole-Paper Critic
# =============================================================================
class GlobalPaperCritic(BaseSkill):
    """
    Comprehensive multi-point audit across the entire manuscript package.
    Evaluates:
      1. Pseudo-replication acknowledgement and effective sample size.
      2. Multiplicity framework consistency (Tukey vs Bonferroni).
      3. AI conversational leakage purge.
      4. Table cleanliness (no raw dictionaries or zeroed slopes).
      5. Reference cross-integrity.
    """

    name = "global_paper_critic"
    description = (
        "Executes a holistic quality and scientific integrity audit across the full paper bundle."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        abstract = context.get("abstract", "")
        introduction = context.get("introduction", "")
        methodology = context.get("methodology", "")
        results_text = context.get("results_text") or context.get("test_interpretation", "")
        discussion = context.get("discussion", "")
        conclusion = context.get("conclusion", "")
        study_facts = context.get("study_facts")
        bibtex = context.get("bibtex", "")
        return self.evaluate(
            abstract=abstract,
            introduction=introduction,
            methodology=methodology,
            results_text=results_text,
            discussion=discussion,
            conclusion=conclusion,
            study_facts=study_facts,
            bibtex=bibtex,
        )

    def evaluate(
        self,
        abstract: str,
        introduction: str,
        methodology: str,
        results_text: str,
        discussion: str,
        conclusion: str,
        study_facts: Optional[StudyFacts] = None,
        bibtex: str = "",
    ) -> CriticEvaluation:
        issues: List[str] = []
        all_text = f"{abstract} {introduction} {methodology} {results_text} {discussion} {conclusion}"

        # 1. AI Conversational Leaks
        leak_patterns = [
            r"Here is (?:a|the) publication-grade",
            r"Note: All statistical outputs adhere strictly",
            r"adhering strictly to the provided study facts",
            r"In this section, I ",
            r"As requested,",
        ]
        for pat in leak_patterns:
            if re.search(pat, all_text, flags=re.IGNORECASE):
                issues.append(f"Contains AI conversational leakage matching pattern: '{pat}'. Must be stripped.")

        # 2. Multiplicity Consistency
        if "bonferroni" in all_text.lower() and "tukey" in all_text.lower():
            if re.search(r"p\s*=\s*0\.0385.*?significant", all_text.lower()):
                # If Bonferroni alpha = 0.0083 was declared, 0.0385 cannot be Bonferroni-significant
                issues.append("Inconsistent multiplicity: declared Bonferroni alpha = 0.0083 but counted p = 0.0385 as significant. Clarify that Tukey HSD (alpha = 0.05) is the primary family-wise post-hoc test.")

        # 3. Pseudo-replication Check
        if study_facts and study_facts.lmm_executed and study_facts.lmm_icc and study_facts.lmm_icc > 0.10:
            if not re.search(r"\b(?:effective\s+sample\s+size|pseudo-replication|between-site\s+variance)\b", all_text, flags=re.IGNORECASE):
                issues.append("Paper fails to explicitly discuss effective sample size (N_eff) and the risk of pseudo-replication given high site-level ICC.")

        # 4. Truncated sentences
        if re.search(r"\bapproximately\s+3\.69\s*$", all_text, flags=re.MULTILINE):
            issues.append("Contains truncated sentence ending with 'approximately 3.69'.")

        # 5. Missing BibTeX keys
        if bibtex:
            defined_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibtex))
            cited_keys = set(re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", all_text))
            for ck in cited_keys:
                for single_key in [k.strip() for k in ck.split(",") if k.strip()]:
                    if single_key not in defined_keys:
                        issues.append(f"In-text citation \\cite{{{single_key}}} missing from references.bib")

        passed = len(issues) == 0
        score = max(0.0, 1.0 - len(issues) * 0.20)
        rev_instr = " | ".join(issues) if issues else None

        return CriticEvaluation(
            passed=passed,
            score=score,
            issues=issues,
            revision_instruction=rev_instr,
            target_section="full_paper",
        )


# =============================================================================
# 5. Cross-Section Consistency Critic (Pairwise Factual Audit)
# =============================================================================
class CrossSectionConsistencyCritic(BaseSkill):
    """
    Pairwise semantic critic that audits factual consistency between section pairs:
      - Introduction <-> Discussion: Contradictory mechanisms, polarity inversions, or false significance claims.
      - Results <-> Discussion: Numerical fidelity, contrast directions, and p-value alignment.
      - Results <-> Abstract: Sample sizes, test statistics, and conclusion alignment.
    """

    name = "cross_section_consistency_critic"
    description = (
        "Audits pairwise section consistency to detect contradictory mechanisms, "
        "directional sign flips, and claims of statistical significance that contradict data."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        sec1_name = context.get("sec1_name", "introduction")
        sec1_text = context.get("sec1_text", "")
        sec2_name = context.get("sec2_name", "discussion")
        sec2_text = context.get("sec2_text", "")
        study_facts = context.get("study_facts")
        return self.evaluate_pair(
            sec1_name=sec1_name,
            sec1_text=sec1_text,
            sec2_name=sec2_name,
            sec2_text=sec2_text,
            study_facts=study_facts,
        )

    def evaluate_section_pair(
        self,
        sec1_name: str,
        sec1_text: str,
        sec2_name: str,
        sec2_text: str,
        study_facts: Optional[StudyFacts] = None,
    ) -> CriticEvaluation:
        return self.evaluate_pair(
            sec1_name=sec1_name,
            sec1_text=sec1_text,
            sec2_name=sec2_name,
            sec2_text=sec2_text,
            study_facts=study_facts,
        )

    def evaluate_pair(
        self,
        sec1_name: str,
        sec1_text: str,
        sec2_name: str,
        sec2_text: str,
        study_facts: Optional[StudyFacts] = None,
    ) -> CriticEvaluation:
        issues: List[str] = []
        if not sec1_text or not sec2_text:
            return CriticEvaluation(passed=True, score=1.0)

        s1_lower = sec1_text.lower()
        s2_lower = sec2_text.lower()

        # 1. Check for mechanism contradiction between Introduction and Discussion
        if sec1_name.lower() == "introduction" and sec2_name.lower() == "discussion":
            # Semantic contradiction checks:
            # Check if one section claims a process does NOT occur while the other claims it DOES
            negation_terms = ["not", "never", "negligible", "does not occur", "no exchange", "infinitesimally slow"]
            pos_terms = ["enhances exchange", "direct exchange", "localized exchange", "drives exchange", "isotopic inheritance"]
            
            for neg in negation_terms:
                if neg in s1_lower:
                    for pos in pos_terms:
                        if pos in s2_lower and ("exchange" in pos or "inheritance" in pos):
                            issues.append(
                                f"Discussion contradicts Introduction mechanism: Introduction establishes that "
                                f"'{neg}' applies to solid-liquid exchange, but Discussion asserts '{pos}'. "
                                f"Discussion must align with the theoretical framework established in Introduction."
                            )
                            break

        # 2. Check for directional polarity flips between raw results and discussion
        if study_facts and study_facts.pairwise_contrasts:
            for pw in study_facts.pairwise_contrasts:
                g1, g2 = pw.group1.lower(), pw.group2.lower()
                # If diff > 0, g2 is higher than g1
                if pw.diff > 0:
                    if f"{g2} is lower" in s2_lower or f"{g2} was lower" in s2_lower or f"{g2} depleted" in s2_lower:
                        issues.append(
                            f"Directional sign flip in {sec2_name}: text claims '{g2}' is lower/depleted relative to '{g1}', "
                            f"but empirical data demonstrates {g2} (mean={pw.mean2:.2f}) is higher than {g1} (mean={pw.mean1:.2f}, diff={pw.diff:+.3f})."
                        )
                elif pw.diff < 0:
                    if f"{g2} is higher" in s2_lower or f"{g2} was higher" in s2_lower or f"{g2} enriched" in s2_lower:
                        issues.append(
                            f"Directional sign flip in {sec2_name}: text claims '{g2}' is higher/enriched relative to '{g1}', "
                            f"but empirical data demonstrates {g2} (mean={pw.mean2:.2f}) is lower than {g1} (mean={pw.mean1:.2f}, diff={pw.diff:+.3f})."
                        )

        # 3. Check for false statistical significance claims
        if study_facts and study_facts.lmm_executed and study_facts.lmm_fixed_effects:
            non_base = [fe for fe in study_facts.lmm_fixed_effects if fe.parameter not in ("const", "Intercept") and not fe.parameter.startswith("Baseline Intercept")]
            if non_base and all(fe.p_value >= 0.05 for fe in non_base):
                if re.search(r"\bstatistically\s+significant\s+(?:association|effect|shift|difference|impact)\b", s2_lower):
                    issues.append(
                        f"{sec2_name} claims statistically significant factor association, but hierarchical model "
                        f"fixed effects are non-significant (all p >= 0.05). Text must state results are inconclusive."
                    )

        # 4. Check for eta-squared misattribution
        if study_facts and study_facts.eta_squared > 0:
            if re.search(r"\b(?:ols|regression|linear\s+model)\s*(?:model\s*)?\(\s*\\eta\^2", s2_lower):
                issues.append(
                    f"{sec2_name} misattributes eta^2 to the OLS regression model. Eta-squared (eta^2 = {study_facts.eta_squared:.4f}) "
                    f"derives from the One-Way ANOVA, whereas regression models report R^2."
                )

        passed = len(issues) == 0
        score = max(0.0, 1.0 - len(issues) * 0.25)
        rev_instr = " | ".join(issues) if issues else None

        return CriticEvaluation(
            passed=passed,
            score=score,
            issues=issues,
            revision_instruction=rev_instr,
            target_section=sec2_name.lower(),
        )


# =============================================================================
# 6. Citation Grounding Critic & Autonomous Literature Retrieval
# =============================================================================
class CitationGroundingCritic(BaseSkill):
    """
    Audits in-text citations against the compiled bibliography (references.bib),
    flags unverified citations and uncited empirical claims, and autonomously
    triggers AcademicLiteratureSearchTool to fetch authentic Crossref papers
    to append to literature_review when missing sources are detected.
    """

    name = "citation_grounding_critic"
    description = (
        "Audits in-text citations against references.bib, flags uncited empirical claims, "
        "and retrieves authentic scholarly sources with DOIs from Crossref."
    )

    def __init__(self, config: Any):
        super().__init__(config)
        from research_agent.tools.search_tools import AcademicLiteratureSearchTool
        self.acad_search = AcademicLiteratureSearchTool()

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        intro_text = context.get("introduction", "")
        disc_text = context.get("discussion", "")
        lit_items = context.get("literature_review") or []
        study_facts = context.get("study_facts")
        return self.evaluate(
            introduction=intro_text,
            discussion=disc_text,
            literature_items=lit_items,
            study_facts=study_facts,
        )

    def evaluate(
        self,
        introduction: str,
        discussion: str,
        literature_items: list[Any],
        bibtex: str = "",
        study_facts: Optional[StudyFacts] = None,
    ) -> CriticEvaluation:
        from research_agent.review import check_citations
        return check_citations(self, introduction, discussion, literature_items)


class WebGroundedCriticSkill(CitationGroundingCritic):
    """Backwards compatibility alias for CitationGroundingCritic."""
    pass


# =============================================================================
# 7. Final Peer-Review Critic (Fresh Context Multi-Perspective Review)
# =============================================================================
class FinalPeerReviewCritic(BaseSkill):
    """
    Isolated, fresh-context final critic that evaluates the complete manuscript draft
    against the empirical StudyFacts / detailed_state_narrative before LaTeX compilation.
    Evaluates:
      1. Complete numerical and sample size consistency with StudyFacts.
      2. No cross-section contradictions in mechanisms, signs, or significance.
      3. Epistemic calibration: wide CIs acknowledged as inconclusive, not 'no effect'.
      4. 100% citation grounding (every claim cited, all keys in references.bib).
    """

    name = "final_peer_review_critic"
    description = (
        "Operates in a fresh context to conduct a rigorous journal peer-review audit of the full paper draft."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> CriticEvaluation:
        manuscript = context.get("manuscript_sections") or {
            "abstract": context.get("abstract", ""),
            "introduction": context.get("introduction", ""),
            "methodology": context.get("methodology", ""),
            "results": context.get("results_text") or context.get("test_interpretation", ""),
            "discussion": context.get("discussion", ""),
            "conclusion": context.get("conclusion", ""),
        }
        study_facts = context.get("study_facts")
        bib = context.get("references_bib", "")
        return self.evaluate_manuscript(
            manuscript_sections=manuscript,
            study_facts=study_facts,
            references_bib=bib,
        )

    def evaluate_manuscript(
        self,
        manuscript_sections: Dict[str, str],
        study_facts: StudyFacts,
        references_bib: str = "",
        source_evidence: Optional[list] = None,
        supplemental_audit: Optional[dict] = None,
    ) -> CriticEvaluation:
        from research_agent.review import review_document
        return review_document(self, manuscript_sections, study_facts, references_bib, source_evidence, supplemental_audit)
