"""
Statistical evaluator and reward scoring engine for LATS tree search.

Extracts p-values from test results, evaluates sample retention tradeoffs,
assesses statistical significance, and computes heuristic + reflection scores.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationScore:
    valid: bool = False
    min_p_value: float = 1.0
    all_p_values: List[float] = field(default_factory=list)
    significant_tests: List[str] = field(default_factory=list)
    total_valid_tests: int = 0
    retention_ratio: float = 1.0
    initial_n: int = 0
    final_n: int = 0
    composite_score: float = 0.0
    reflection: str = ""

    # Multiplicity & Garden-of-Forking-Paths Control
    k_paths_explored: int = 1
    bonferroni_alpha: float = 0.05
    fdr_threshold: float = 0.05
    is_significant_after_bonferroni: bool = False
    is_significant_after_fdr: bool = False
    fdr_adjusted_min_p: float = 1.0
    fdr_adjusted_p_values: List[Tuple[str, float]] = field(default_factory=list)


class LATSEvaluator:
    """Evaluates statistical test outputs and scores trajectories."""

    P_VALUE_KEYS = {
        "p",
        "p_value",
        "p-value",
        "p_val",
        "pval",
        "pr(>|t|)",
        "pr(>f)",
        "p-val",
        "p(>|z|)",
    }

    @classmethod
    def extract_p_values(cls, data: Any) -> List[Tuple[str, float]]:
        """
        Recursively extracts all p-values from test results dictionary.
        Returns a list of tuples: (context_name, p_value).
        """
        results: List[Tuple[str, float]] = []

        def add(value: Any, path: str):
            if isinstance(value, bool):
                return
            try:
                p = float(value)
                if math.isfinite(p) and 0 <= p <= 1:
                    results.append((path, p))
            except (ValueError, TypeError):
                pass

        def _traverse(node: Any, path: str, p_map: bool = False):
            if isinstance(node, dict):
                for k, v in node.items():
                    key = str(k).strip().lower()
                    if key.startswith('_'):
                        continue  # telemetry is not an additional hypothesis test
                    child = f"{path}.{k}" if path else str(k)
                    if key in cls.P_VALUE_KEYS:
                        add(v, path or str(k))
                    elif key in ("p_values", "pvalues", "p-values"):
                        _traverse(v, child, p_map=True)
                    elif isinstance(v, (dict, list, tuple)):
                        _traverse(v, child, p_map)
                    elif p_map and key not in ("const", "intercept") and not key.startswith("baseline intercept"):
                        add(v, child)
            elif isinstance(node, (list, tuple)):
                for idx, item in enumerate(node):
                    _traverse(item, f"{path}[{idx}]", p_map)

        _traverse(data, "")
        return results

    @classmethod
    def format_p_value(cls, p: float) -> str:
        """Formats p-values into clean scientific notation or bounds."""
        if p <= 0.0 or p < 1e-15:
            return "p < 1e-15"
        elif p < 0.0001:
            return f"p={p:.3e}"
        else:
            return f"p={p:.4g}"

    @classmethod
    def get_minimum_p_value(cls, data: Any) -> float:
        """Returns the lowest p-value found, or 1.0 if none exist."""
        extracted = cls.extract_p_values(data)
        if not extracted:
            return 1.0
        return min(p for _, p in extracted)

    @classmethod
    def get_significant_tests(cls, data: Any, alpha: float = 0.05) -> List[str]:
        """Returns list of test names whose p-value is below alpha."""
        extracted = cls.extract_p_values(data)
        return [f"{name} ({cls.format_p_value(p)})" for name, p in extracted if p < alpha]

    @classmethod
    def calculate_bonferroni(
        cls,
        p_values: List[float],
        alpha: float = 0.05,
        n_comparisons: Optional[int] = None,
    ) -> Tuple[float, List[bool]]:
        """
        Computes Bonferroni-corrected alpha threshold and per-test significance.
        Returns: (bonferroni_alpha, [is_sig_test_1, ...])
        """
        m = n_comparisons if (n_comparisons is not None and n_comparisons > 0) else len(p_values)
        if m <= 0:
            return alpha, []
        bonf_alpha = alpha / float(m)
        is_sig = [p < bonf_alpha for p in p_values]
        return bonf_alpha, is_sig

    @classmethod
    def calculate_benjamini_hochberg(
        cls,
        named_p_values: List[Tuple[str, float]],
        fdr_rate: float = 0.05,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        Benjamini-Hochberg (BH) False Discovery Rate procedure.
        Computes monotonic FDR-adjusted p-values (q-values) and returns:
          - [(name, q_value), ...]
          - list of names of significant tests under FDR control
        """
        m = len(named_p_values)
        if m == 0:
            return [], []

        # Sort tests by p-value ascending
        sorted_indices = sorted(range(m), key=lambda i: named_p_values[i][1])
        sorted_pairs = [named_p_values[i] for i in sorted_indices]

        # Calculate raw adjusted p-values: q_i = p_i * m / rank (1-based rank)
        raw_q = []
        for rank_1b, (name, p) in enumerate(sorted_pairs, start=1):
            raw_q.append(min(1.0, p * m / float(rank_1b)))

        # Enforce monotonicity backwards: q_i = min(q_i, q_{i+1})
        monotonic_q = [0.0] * m
        current_min = 1.0
        for i in reversed(range(m)):
            current_min = min(current_min, raw_q[i])
            monotonic_q[i] = current_min

        # Re-map back to original order
        adjusted_results: List[Tuple[str, float]] = [("", 1.0)] * m
        significant_names: List[str] = []
        for rank_idx, orig_idx in enumerate(sorted_indices):
            name, _ = sorted_pairs[rank_idx]
            q_val = monotonic_q[rank_idx]
            adjusted_results[orig_idx] = (name, q_val)
            if q_val <= fdr_rate:
                significant_names.append(name)

        return adjusted_results, significant_names

    @classmethod
    def apply_multiplicity_correction(
        cls,
        trajectories: List[Any],
        alpha: float = 0.05,
    ):
        """
        Applies family-wise Bonferroni and Benjamini-Hochberg FDR multiplicity
        corrections across all candidate trajectories in a search tree.
        Mutates trajectory multiplicity attributes in-place.
        """
        k = len(trajectories)
        if k == 0:
            return

        # Correct the entire family, not just a selected minimum from each path.
        family = [(f"{t.id}:{name}", p)
                  for t in trajectories for name, p in cls.extract_p_values(t.test_results)]
        bonf_alpha = alpha / max(1, len(family))
        adjusted, _ = cls.calculate_benjamini_hochberg(family, fdr_rate=alpha)
        for t in trajectories:
            own = [p for name, p in family if name.startswith(f"{t.id}:")]
            q_values = [q for name, q in adjusted if name.startswith(f"{t.id}:")]
            t.k_paths_explored = k
            t.bonferroni_alpha = bonf_alpha
            t.is_significant_after_bonferroni = any(p < bonf_alpha for p in own)
            t.fdr_adjusted_min_p = min(q_values, default=1.0)
            t.is_significant_after_fdr = bool(q_values) and t.fdr_adjusted_min_p <= alpha
            t.reflection = (
                f"Exploratory sensitivity analysis: {len(own)} tests in this path; "
                f"{len(family)} tests across {k} paths. "
                f"Bonferroni significance: {t.is_significant_after_bonferroni}; "
                f"minimum FDR q={t.fdr_adjusted_min_p:.4g}. "
                "Route ranking uses sample retention, not statistical significance."
            )

    @classmethod
    def evaluate(
        cls,
        test_results: Dict[str, Any],
        initial_n: int,
        final_n: int,
        alpha: float = 0.05,
        cleaning_strategy: str = "",
        hypothesis: str = "",
        k_paths_explored: int = 1,
    ) -> EvaluationScore:
        """
        Scores a candidate (Cleaning Option + Hypothesis) outcome with multiplicity adjustments.

        Score Formula:
          Composite = retained fraction for a valid analysis (independent of p-values)
        """
        if not test_results or "error" in test_results:
            return EvaluationScore(
                valid=False,
                composite_score=0.0,
                reflection=f"Test failed or returned error: {test_results.get('error', 'empty results')}",
            )

        extracted = cls.extract_p_values(test_results)
        if not extracted:
            return EvaluationScore(
                valid=False,
                composite_score=0.05,
                reflection="Tests executed but no numerical p-values were extracted.",
            )

        p_vals = [p for _, p in extracted]
        min_p = min(p_vals)
        total_tests = len(p_vals)

        # Multiplicity corrections within trajectory
        bonf_alpha, bonf_sig_flags = cls.calculate_bonferroni(
            p_vals, alpha=alpha, n_comparisons=total_tests * max(1, k_paths_explored)
        )
        fdr_adjusted, fdr_sig_names = cls.calculate_benjamini_hochberg(extracted, fdr_rate=alpha)
        fdr_min_p = min(q for _, q in fdr_adjusted) if fdr_adjusted else 1.0

        is_sig_bonf = any(bonf_sig_flags)
        is_sig_fdr = len(fdr_sig_names) > 0

        # Count nominally significant tests (at alpha level)
        sig_tests = []
        for name, p in extracted:
            if p < alpha:
                sig_tests.append(f"{name} ({cls.format_p_value(p)})")

        retention = (final_n / initial_n) if initial_n > 0 else 0.0
        # Non-significance is a valid outcome. Never reward p-value minimization.
        composite = max(0.0, min(1.0, retention))

        # Reflection generation with multiplicity audit
        if is_sig_bonf:
            sig_status = f"Strong significance: survives Bonferroni (min p={cls.format_p_value(min_p)} < {bonf_alpha:.4g})"
        elif is_sig_fdr:
            sig_status = f"Significant under FDR control (min p={cls.format_p_value(min_p)}, FDR q={fdr_min_p:.4g})"
        elif min_p < alpha:
            sig_status = f"Nominally significant (min p={cls.format_p_value(min_p)}) but fails family-wise multiplicity control (Bonferroni alpha={bonf_alpha:.4g})"
        else:
            sig_status = f"Not significant (min p={cls.format_p_value(min_p)})"

        reflection = (
            f"Strategy '{cleaning_strategy}': retained {final_n}/{initial_n} rows ({retention*100:.1f}%). "
            f"{sig_status} across {len(sig_tests)}/{total_tests} test metrics. "
            f"Overall trajectory score: {composite:.3f}."
        )

        return EvaluationScore(
            valid=True,
            min_p_value=min_p,
            all_p_values=p_vals,
            significant_tests=sig_tests,
            total_valid_tests=total_tests,
            retention_ratio=retention,
            initial_n=initial_n,
            final_n=final_n,
            composite_score=composite,
            reflection=reflection,
            k_paths_explored=k_paths_explored,
            bonferroni_alpha=bonf_alpha,
            fdr_threshold=alpha,
            is_significant_after_bonferroni=is_sig_bonf,
            is_significant_after_fdr=is_sig_fdr,
            fdr_adjusted_min_p=fdr_min_p,
            fdr_adjusted_p_values=fdr_adjusted,
        )
