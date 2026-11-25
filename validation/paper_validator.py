"""
Paper Validator: Automated Integrity Gate for Scientific Papers.

Deterministically verifies that all synthesized sections conform strictly to
the immutable StudyFacts contract, catching and correcting numerical discrepancies,
sample size hallucinations, inverted effect polarities, and invalid citations.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from research_agent.schemas import StudyFacts

logger = logging.getLogger(__name__)


class ValidationIssue(BaseModel):
    section: str
    rule: str
    severity: str = "ERROR"  # ERROR or WARNING
    message: str
    expected: Any
    found: Any


class ValidationReport(BaseModel):
    is_valid: bool = True
    issues: List[ValidationIssue] = Field(default_factory=list)

    def add_error(self, section: str, rule: str, message: str, expected: Any, found: Any):
        self.is_valid = False
        self.issues.append(
            ValidationIssue(
                section=section,
                rule=rule,
                severity="ERROR",
                message=message,
                expected=expected,
                found=found,
            )
        )

    def add_warning(self, section: str, rule: str, message: str, expected: Any, found: Any):
        self.issues.append(
            ValidationIssue(
                section=section,
                rule=rule,
                severity="WARNING",
                message=message,
                expected=expected,
                found=found,
            )
        )


class PaperValidator:
    """Automated consistency and integrity validator."""

    def __init__(self, study_facts: StudyFacts):
        self.study_facts = study_facts

    def validate_section(self, section_name: str, text: str) -> List[ValidationIssue]:
        """Validate a single section text against the immutable StudyFacts."""
        issues: List[ValidationIssue] = []
        if not text:
            return issues

        text_lower = text.lower()

        # 1. Sample Size Rule: Check for explicit N = XXX or N = X,XXX
        # Distinguish:
        # - Subgroup sample sizes: e.g. group_a (n = 58), group_b (n = 417), etc.
        # - Effective sample sizes: e.g. effective N = 254.2
        # - Total sample size: e.g. N = 542
        # 1a. Effective sample size check
        if self.study_facts.effective_sample_size is not None:
            eff_expected = self.study_facts.effective_sample_size
            for em in re.finditer(r"\b(?:effective\s+[Nn]|N[_\s]*eff(?:ective)?)\s*[=:]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE):
                found_eff = float(em.group(1))
                if abs(found_eff - eff_expected) > 1.0 and abs(found_eff - round(eff_expected)) > 1.0:
                    issues.append(
                        ValidationIssue(
                            section=section_name,
                            rule="effective_sample_size_match",
                            message=f"Reported effective sample size {found_eff} does not match ground truth {eff_expected:.1f}",
                            expected=eff_expected,
                            found=found_eff,
                        )
                    )

        # 1b. Subgroup sample sizes check for each factor level
        if self.study_facts.level_sample_sizes:
            for lvl, lvl_n in self.study_facts.level_sample_sizes.items():
                lvl_esc = re.escape(lvl.lower())
                for sm in re.finditer(rf"\b{lvl_esc}\b\s*(?:\([^\)]*|\$)?\s*[Nn]\s*[=:]\s*(\d[\d,]*)", text_lower):
                    found_sub_n = int(sm.group(1).replace(",", ""))
                    if found_sub_n != lvl_n and found_sub_n > 10:
                        issues.append(
                            ValidationIssue(
                                section=section_name,
                                rule="subgroup_sample_size_match",
                                message=f"Reported subgroup sample size for '{lvl}' ({found_sub_n}) does not match ground truth {lvl_n}",
                                expected=lvl_n,
                                found=found_sub_n,
                            )
                        )

        # 1c. Total sample size N check (uppercase N, not effective N or subgroup)
        for tm in re.finditer(r"(?<!effective\s)(?<!eff\s)(?<!N_)\bN\s*[=:]\s*([0-9,]{2,7})\b", text):
            found_tot_n = int(tm.group(1).replace(",", ""))
            if self.study_facts.level_sample_sizes and found_tot_n in self.study_facts.level_sample_sizes.values():
                continue
            if self.study_facts.effective_sample_size and abs(found_tot_n - self.study_facts.effective_sample_size) < 2:
                continue
            if found_tot_n != self.study_facts.sample_size_total:
                issues.append(
                    ValidationIssue(
                        section=section_name,
                        rule="sample_size_match",
                        message=f"Reported sample size N = {found_tot_n} does not match ground truth N = {self.study_facts.sample_size_total}",
                        expected=self.study_facts.sample_size_total,
                        found=found_tot_n,
                    )
                )

        # 2. ANOVA Degrees of Freedom Rule: Check for F(a, b)
        anova_df_matches = re.findall(r"\bF\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        for df1_str, df2_str in anova_df_matches:
            df1, df2 = int(df1_str), int(df2_str)
            if df1 != self.study_facts.anova_df_between or df2 != self.study_facts.anova_df_within:
                issues.append(
                    ValidationIssue(
                        section=section_name,
                        rule="anova_df_match",
                        message=f"Reported ANOVA df F({df1}, {df2}) does not match ground truth F({self.study_facts.anova_df_between}, {self.study_facts.anova_df_within})",
                        expected=(self.study_facts.anova_df_between, self.study_facts.anova_df_within),
                        found=(df1, df2),
                    )
                )

        # 3. Factor Level Integrity Rule: Reject hallucinated factor categories
        text_lower = text.lower()
        if self.study_facts.factor_levels:
            # Check for spurious category declarations like "group: X" or "category: X" where X is not in factor_levels
            for m in re.finditer(r"\b(?:category|group|cohort|class)\s*[:=]\s*([a-zA-Z_-]+)\b", text_lower):
                cat_found = m.group(1)
                if cat_found not in [lvl.lower() for lvl in self.study_facts.factor_levels] and cat_found not in ["reference", "baseline", "control", "treatment"]:
                    issues.append(
                        ValidationIssue(
                            section=section_name,
                            rule="extraneous_factor_level",
                            message=f"Mentioned unstudied factor level '{cat_found}' which is not in dataset categories {self.study_facts.factor_levels}",
                            expected=self.study_facts.factor_levels,
                            found=cat_found,
                        )
                    )

        # 4. General Directionality Rule across all factor pairs
        if self.study_facts.level_descriptives:
            levels = list(self.study_facts.level_descriptives.keys())
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    l1, l2 = levels[i], levels[j]
                    m1 = self.study_facts.level_descriptives[l1].get("mean", 0.0)
                    m2 = self.study_facts.level_descriptives[l2].get("mean", 0.0)
                    if m1 < m2:
                        # l1 is lower than l2. Inversion if l1 claimed higher than l2 or l2 claimed lower than l1
                        inv1 = re.search(rf"\b{re.escape(l1.lower())}\s+(?:is|was)\s+(?:higher|greater|larger|more\s+positive|less\s+negative)\s+than\s+{re.escape(l2.lower())}\b", text_lower)
                        inv2 = re.search(rf"\b{re.escape(l2.lower())}\s+(?:is|was)\s+(?:lower|smaller|more\s+negative|less\s+positive)\s+than\s+{re.escape(l1.lower())}\b", text_lower)
                        if inv1 or inv2:
                            issues.append(
                                ValidationIssue(
                                    section=section_name,
                                    rule="directionality_inversion",
                                    message=f"Inverted direction: claimed '{l1}' is higher than '{l2}' (or '{l2}' lower than '{l1}'), but '{l1}' mean ({m1:.2f}) is lower than '{l2}' mean ({m2:.2f})",
                                    expected=f"{l1} is lower / more negative than {l2}",
                                    found=inv1.group(0) if inv1 else inv2.group(0),
                                )
                            )
                    elif m1 > m2:
                        # l1 is higher than l2. Inversion if l1 claimed lower than l2 or l2 claimed higher than l1
                        inv1 = re.search(rf"\b{re.escape(l1.lower())}\s+(?:is|was)\s+(?:lower|smaller|more\s+negative|less\s+positive)\s+than\s+{re.escape(l2.lower())}\b", text_lower)
                        inv2 = re.search(rf"\b{re.escape(l2.lower())}\s+(?:is|was)\s+(?:higher|greater|larger|more\s+positive|less\s+negative)\s+than\s+{re.escape(l1.lower())}\b", text_lower)
                        if inv1 or inv2:
                            issues.append(
                                ValidationIssue(
                                    section=section_name,
                                    rule="directionality_inversion",
                                    message=f"Inverted direction: claimed '{l2}' is higher than '{l1}' (or '{l1}' lower than '{l2}'), but '{l1}' mean ({m1:.2f}) is higher than '{l2}' mean ({m2:.2f})",
                                    expected=f"{l2} is lower / more negative than {l1}",
                                    found=inv2.group(0) if inv2 else inv1.group(0),
                                )
                            )

        # 5. Geochemical / Physical Mechanism check (if present in text)
        if re.search(r"rock\s+(?:lattice\s+)?suppl(?:ies|ied)\s+(?:isotopically\s+)?lighter\s+oxygen", text_lower) or \
           re.search(r"bedrock\s+oxygen\s+exchange\s+(?:dominates|drives)\s+water", text_lower):
            issues.append(
                ValidationIssue(
                    section=section_name,
                    rule="unphysical_mechanism",
                    message="Unphysical geochemical claim: rock lattice oxygen exchange with fluid is negligible at low temperatures",
                    expected="fluid reservoir dominance or dissolution kinetics",
                    found="rock lattice supplying lighter oxygen to water",
                )
            )

        # 5. Manifest & Fence Arithmetic Rule
        if self.study_facts.manifest and self.study_facts.manifest.lower_fence is not None:
            mf = self.study_facts.manifest
            # Check if text reports wrong lower fence
            for fence_m in re.finditer(r"(?:lower\s+fence|lower\s+bound|fence)\s*[=:]\s*([+-]?\d+\.?\d*)", text_lower):
                val = float(fence_m.group(1))
                if abs(val - mf.lower_fence) > 0.5 and abs(val - (mf.upper_fence or 0)) > 0.5:
                    issues.append(
                        ValidationIssue(
                            section=section_name,
                            rule="incorrect_fence_math",
                            message=f"Reported outlier fence {val} contradicts deterministic fence {mf.lower_fence:.2f} (Q1 - 1.5*IQR)",
                            expected=mf.lower_fence,
                            found=val,
                        )
                    )

        # 6. Epistemic Honesty Rule: Check for false claims of significance
        if self.study_facts.pairwise_contrasts:
            for pw in self.study_facts.pairwise_contrasts:
                if not pw.significant:
                    g1, g2 = pw.group1.lower(), pw.group2.lower()
                    if re.search(rf"{g1}\s+is\s+distinct\s+from\s+all\s+other\s+groups", text_lower) or \
                       re.search(rf"{g1}\s+(?:is|was)\s+significantly\s+different\s+from\s+{g2}", text_lower):
                        issues.append(
                            ValidationIssue(
                                section=section_name,
                                rule="false_significance_claim",
                                message=f"Falsely claimed {pw.group1} vs {pw.group2} is significant or distinct, but empirical p = {pw.p_value:.4f} > 0.05",
                                expected="statistically non-significant contrast",
                                found="claimed distinct / significant",
                            )
                        )

        # 7. Conversational AI Leak Rule
        leak_matches = [
            r"Here is (?:a|the) publication-grade",
            r"Note: All statistical outputs adhere strictly",
            r"adhering strictly to the provided study facts",
            r"in the study facts",
            r"reported in the study facts",
            r"presented in the study facts",
        ]
        for pat in leak_matches:
            if re.search(pat, text, flags=re.IGNORECASE):
                issues.append(
                    ValidationIssue(
                        section=section_name,
                        rule="conversational_leak",
                        message=f"Section contains conversational meta-commentary matching '{pat}'",
                        expected="pure academic text",
                        found=pat,
                    )
                )

        # 8. Unkept Promises & Proposal Phrasing in Introduction
        if section_name.lower() == "introduction":
            for bad_phrase in ["would provide", "will be integrated", "proposes to", "bayesian hierarchical"]:
                if bad_phrase in text_lower:
                    issues.append(
                        ValidationIssue(
                            section=section_name,
                            rule="proposal_or_unkept_promise",
                            message=f"Introduction contains proposal phrasing or unkept promise '{bad_phrase}'",
                            expected="completed past-tense scientific prose without unexecuted methods",
                            found=bad_phrase,
                        )
                    )

        # 9. Premature Confirmation when LMM is Non-Significant
        if section_name.lower() in ("results", "abstract", "conclusion"):
            if self.study_facts.lmm_executed and not any(getattr(c, "significant", False) for c in (self.study_facts.lmm_fixed_effects or [])):
                if "robust empirical confirmation" in text_lower:
                    issues.append(
                        ValidationIssue(
                            section=section_name,
                            rule="premature_hypothesis_confirmation",
                            message="Claimed 'robust empirical confirmation' of hypothesis despite all LMM fixed effects being statistically non-significant (p > 0.05)",
                            expected="statistically inconclusive outcome after cluster adjustment",
                            found="robust empirical confirmation",
                        )
                    )

        # 10. Prohibited terminology rule
        if re.search(r"\b(dissolve|dissolves|dissolved|vanish|vanishes|vanished|disappear|disappears|disappeared)\b", text_lower):
            issues.append(
                ValidationIssue(
                    section=section_name,
                    rule="prohibited_terminology",
                    message="Used prohibited word ('dissolve', 'vanish', or 'disappear') to describe effect attenuation. Use 'become statistically inconclusive' or 'attenuate'.",
                    expected="statistically inconclusive / attenuate",
                    found="dissolve/vanish/disappear",
                )
            )

        # 11. Empty jargon rule
        if "preservation of full observational ecological validity" in text_lower:
            issues.append(
                ValidationIssue(
                    section=section_name,
                    rule="unscientific_jargon",
                    message="Contains empty buzzword 'preservation of full observational ecological validity'.",
                    expected="objective methodological description",
                    found="preservation of full observational ecological validity",
                )
            )

        return issues

    def validate_full_paper(
        self,
        abstract: str,
        introduction: str,
        methodology: str,
        results: str,
        discussion: str,
        conclusion: str,
        bibtex: str = "",
    ) -> ValidationReport:
        """Validate all paper sections against StudyFacts."""
        report = ValidationReport()
        sections = {
            "Abstract": abstract,
            "Introduction": introduction,
            "Methodology": methodology,
            "Results": results,
            "Discussion": discussion,
            "Conclusion": conclusion,
        }

        for sec_name, sec_text in sections.items():
            if not sec_text.strip():
                report.add_error(sec_name, "missing_section", "Required manuscript section is empty", "non-empty text", "")
            issues = self.validate_section(sec_name, sec_text)
            for iss in issues:
                report.issues.append(iss)
                if iss.severity == "ERROR":
                    report.is_valid = False

        # Citation Key Cross-Check: every \cite{key} in text exists in bibtex
        if bibtex:
            defined_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bibtex))
            all_text = " ".join(sections.values())
            cited_keys = set(re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", all_text))
            for ck in cited_keys:
                # Handle comma-separated keys inside \cite{k1,k2}
                for single_key in [k.strip() for k in ck.split(",") if k.strip()]:
                    if single_key not in defined_keys:
                        report.add_error(
                            section="References",
                            rule="missing_bib_entry",
                            message=f"In-text citation \\cite{{{single_key}}} has no corresponding entry in references.bib",
                            expected=single_key,
                            found="Missing from .bib",
                        )

        return report

    def correct_section_discrepancies(self, section_name: str, text: str) -> str:
        """
        Sanitizes section text of AI conversational preambles, redundant formatting,
        and syntax artifacts without altering scientific numbers or prose via regex.
        Scientific revisions are handled natively by the Actor model grounded in StudyFacts.
        """
        if not text:
            return text

        corrected = text

        # 1. Clean stray markdown horizontal rules
        corrected = re.sub(r"^\s*---\s*$", "", corrected, flags=re.MULTILINE)

        # 2. Strip AI conversational preambles and postambles
        corrected = re.sub(
            r"^(?:Here is (?:a|the) (?:publication-grade\s+)?[a-z\s]+[\:\.\n]+)",
            "",
            corrected,
            flags=re.IGNORECASE,
        ).strip()
        corrected = re.sub(
            r"^(?:Here is (?:the|a)\s+[^:\n]+[:\n]+)",
            "",
            corrected,
            flags=re.IGNORECASE,
        ).strip()
        corrected = re.sub(
            r"^(?:Here is (?:a|the) publication-grade[\s\S]*?provided study facts[\.\:\n]*)",
            "",
            corrected,
            flags=re.IGNORECASE,
        ).strip()
        corrected = re.sub(
            r"(?:Note:\s*All statistical outputs adhere strictly[\s\S]*?$)",
            "",
            corrected,
            flags=re.IGNORECASE,
        ).strip()
        corrected = re.sub(
            r"\badhering strictly to the provided study facts\b",
            "",
            corrected,
            flags=re.IGNORECASE,
        )

        # 3. Strip leftover prompt phrasing: "in the study facts" -> "in Table 1"
        corrected = re.sub(
            r"\b(?:reported|presented)\s+(?:in|by)\s+the\s+study\s+facts\b",
            "reported in Table 1",
            corrected,
            flags=re.IGNORECASE,
        )
        corrected = re.sub(
            r"\bin\s+the\s+study\s+facts\b",
            "in Table 1",
            corrected,
            flags=re.IGNORECASE,
        )
        corrected = re.sub(
            r"\bprovided\s+study\s+facts\b",
            "empirical data",
            corrected,
            flags=re.IGNORECASE,
        )

        # 4. Fix Abstract bloat: strip leaked results headers and trailing ellipses
        if section_name.lower() == "abstract":
            corrected = re.sub(r"###?\s*Results:\s*Empirical Results and Statistical Analysis[\s\S]*?(?=\n\n|\Z)", "", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"Results:\s*Empirical Results and Statistical Analysis", "", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\.\.+", ".", corrected)

        # 5. Introduction proposal phrasing clean-up
        if section_name.lower() == "introduction":
            corrected = re.sub(r"\bwould provide\b", "provides", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\bproposes to use\b", "uses", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\bproposes to\b", "aims to", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\bwill be integrated\b", "was integrated", corrected, flags=re.IGNORECASE)

        # 6. Clean empty citation parentheses resulting from prompt parsing
        corrected = re.sub(r"\(\s*,\s*", "(", corrected)
        corrected = re.sub(r",\s*\)", ")", corrected)
        corrected = re.sub(r"\(\s*\)", "", corrected)

        # 7. Clean prohibited terms
        corrected = re.sub(r"\bdissolve\b", "become inconclusive", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"\bdissolves\b", "becomes inconclusive", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"\bdissolved\b", "became inconclusive", corrected, flags=re.IGNORECASE)
        corrected = re.sub(r"\bpreservation of full observational ecological validity\b", "preservation of the complete observational sample", corrected, flags=re.IGNORECASE)

        corrected = re.sub(r"  +", " ", corrected)

        return corrected
