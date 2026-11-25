"""
Pool of diverse data cleaning strategies for LATS tree search.

Implements multiple distinct data preparation pathways:
  1. Standard IQR (1.5x) Outlier Filtering
  2. Conservative IQR (3.0x) Severe Outlier Filtering
  3. No Outlier Removal (Preserve Full Distribution)
  4. Quantile Winsorization (Clipping without dropping rows)
  5. Z-Score Filtering (|z| <= 3.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from research_agent.tools.data_tools import (
    clean_na,
    convert_to_categorical,
    convert_to_numeric,
    determine_type_of_variable,
)

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    cleaned_df: pd.DataFrame
    strategy_name: str
    description: str
    initial_n: int
    final_n: int
    retention_ratio: float
    removed_outliers: int

    @property
    def initial_sample_size(self) -> int:
        return self.initial_n

    @property
    def retained_sample_size(self) -> int:
        return self.final_n



class CleaningStrategy:
    """A parameterized data cleaning option."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[[pd.DataFrame, str, List[str], List[str]], pd.DataFrame],
    ):
        self.name = name
        self.description = description
        self.func = func

    def clean(
        self,
        df: pd.DataFrame,
        dependent_variable: str,
        independent_variables: List[str],
        covariates: Optional[List[str]] = None,
    ) -> CleaningResult:
        covs = covariates or []
        raw_total_n = len(df)

        # Preserve an audit of the actual cleaning operation, separate from missingness.
        baseline, _ = _base_na_and_typecast(df, dependent_variable, independent_variables, covs)
        cleaned_df = self.func(df, dependent_variable, independent_variables, covs)
        final_n = len(cleaned_df)
        outlier_removal = self.name in {"iqr_standard", "iqr_conservative", "zscore_filter"}
        outliers = max(0, len(baseline) - final_n) if outlier_removal else 0
        audit = {
            "raw_sample_size": raw_total_n,
            "final_sample_size": final_n,
            "missing_rows_removed": max(0, raw_total_n - final_n - outliers),
            "outliers_removed": outliers,
            "strategy": self.name,
            "filters": cleaned_df.attrs.get("outlier_filters", []),
            "transformations": cleaned_df.attrs.get("transformations", {}),
        }
        cleaned_df.attrs["cleaning_audit"] = audit
        baseline_n = raw_total_n
        retention = (final_n / raw_total_n) if raw_total_n else 0.0

        return CleaningResult(
            cleaned_df=cleaned_df,
            strategy_name=self.name,
            description=self.description,
            initial_n=baseline_n,
            final_n=final_n,
            retention_ratio=retention,
            removed_outliers=outliers,
        )


def _base_na_and_typecast(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    """Helper to clean NAs and typecast relevant columns."""
    relevant_cols = [dependent_variable] + independent_variables + covariates
    valid_indices = set(df.index)

    for col in relevant_cols:
        if col in df.columns:
            cleaned_s = clean_na(df[col])
            valid_indices = valid_indices.intersection(set(cleaned_s.index))

    df_clean = df.loc[df.index.isin(valid_indices)].copy()
    df_clean.attrs = {}
    if df_clean.empty:
        return df_clean, relevant_cols

    for col in relevant_cols:
        if col not in df_clean.columns:
            continue
        col_type = determine_type_of_variable(col, df_clean)
        if col_type == "numeric":
            df_clean[col] = convert_to_numeric(df_clean[col])
            df_clean[col + "_original"] = df_clean[col]
        elif col_type in ("categorical", "binary"):
            df_clean[col] = convert_to_categorical(df_clean[col])
            df_clean[col + "_original"] = df_clean[col]

    df_clean = df_clean.dropna(subset=[c for c in relevant_cols if c in df_clean.columns])
    df_clean = df_clean.reset_index(drop=True)
    return df_clean, relevant_cols


# ---------------------------------------------------------------------------
# Strategy 1: Standard IQR (1.5x)
# ---------------------------------------------------------------------------
def clean_iqr_standard(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> pd.DataFrame:
    df_clean, relevant_cols = _base_na_and_typecast(df, dependent_variable, independent_variables, covariates)
    if df_clean.empty:
        return df_clean

    for col in relevant_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                before = len(df_clean)
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)].copy()
                df_clean.attrs.setdefault("outlier_filters", []).append({
                    "variable": col, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                    "lower_fence": float(lower), "upper_fence": float(upper),
                    "removed_count": before - len(df_clean),
                })

    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy 2: Conservative IQR (3.0x - severe outliers only)
# ---------------------------------------------------------------------------
def clean_iqr_conservative(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> pd.DataFrame:
    df_clean, relevant_cols = _base_na_and_typecast(df, dependent_variable, independent_variables, covariates)
    if df_clean.empty:
        return df_clean

    for col in relevant_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                before = len(df_clean)
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)].copy()
                df_clean.attrs.setdefault("outlier_filters", []).append({
                    "variable": col, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                    "lower_fence": float(lower), "upper_fence": float(upper),
                    "removed_count": before - len(df_clean),
                })

    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy 3: No Outlier Removal (Preserve Full Population)
# ---------------------------------------------------------------------------
def clean_no_outlier_removal(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> pd.DataFrame:
    df_clean, _ = _base_na_and_typecast(df, dependent_variable, independent_variables, covariates)
    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy 4: Winsorization (Clip extremes, 0 drops)
# ---------------------------------------------------------------------------
def clean_winsorization(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> pd.DataFrame:
    df_clean, relevant_cols = _base_na_and_typecast(df, dependent_variable, independent_variables, covariates)
    if df_clean.empty:
        return df_clean

    for col in relevant_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            lower = df_clean[col].quantile(0.02)
            upper = df_clean[col].quantile(0.98)
            df_clean[col + "_original"] = df_clean[col].copy()
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            df_clean.attrs.setdefault("transformations", {})[col] = f"Winsorized to [{lower}, {upper}] (2nd/98th percentiles)"

    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy 5: Z-Score Filtering (|z| <= 3.0)
# ---------------------------------------------------------------------------
def clean_zscore(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> pd.DataFrame:
    df_clean, relevant_cols = _base_na_and_typecast(df, dependent_variable, independent_variables, covariates)
    if df_clean.empty:
        return df_clean

    for col in relevant_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            std = df_clean[col].std()
            if std > 0:
                mean = df_clean[col].mean()
                z = (df_clean[col] - mean).abs() / std
                before = len(df_clean)
                df_clean = df_clean[z <= 3.0].copy()
                df_clean.attrs.setdefault("outlier_filters", []).append({
                    "variable": col, "mean": float(mean), "std": float(std),
                    "lower_fence": float(mean - 3 * std), "upper_fence": float(mean + 3 * std),
                    "removed_count": before - len(df_clean),
                })

    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy 6: Median / Mode Covariate Imputation (High Retention)
# ---------------------------------------------------------------------------
def clean_median_imputation(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: List[str],
    covariates: List[str],
) -> pd.DataFrame:
    """
    Imputes missing numeric covariates with sample median, categorical with mode.
    Preserves all observations where primary independent & dependent variables exist.
    """
    relevant_cols = [dependent_variable] + independent_variables + (covariates or [])
    primary_cols = [dependent_variable] + independent_variables

    # Drop rows missing primary variables (cannot impute focal outcome or exposure)
    valid_indices = set(df.index)
    for col in primary_cols:
        if col in df.columns:
            cleaned_s = clean_na(df[col])
            valid_indices = valid_indices.intersection(set(cleaned_s.index))

    df_clean = df.loc[df.index.isin(valid_indices)].copy()
    if df_clean.empty:
        return df_clean

    for col in relevant_cols:
        if col not in df_clean.columns:
            continue
        col_type = determine_type_of_variable(col, df_clean)
        if col_type == "numeric":
            df_clean[col] = convert_to_numeric(df_clean[col])
            if col in (covariates or []) and df_clean[col].isnull().any():
                med = df_clean[col].median()
                if not pd.isna(med):
                    df_clean[col] = df_clean[col].fillna(med)
            df_clean[col + "_original"] = df_clean[col]
        elif col_type in ("categorical", "binary"):
            df_clean[col] = convert_to_categorical(df_clean[col])
            if col in (covariates or []) and df_clean[col].isnull().any():
                mode_vals = df_clean[col].mode()
                if not mode_vals.empty:
                    df_clean[col] = df_clean[col].fillna(mode_vals.iloc[0])
            df_clean[col + "_original"] = df_clean[col]

    df_clean = df_clean.dropna(subset=[c for c in primary_cols if c in df_clean.columns])
    return df_clean.reset_index(drop=True)


class CleanerPool:
    """Manages the candidate cleaning strategies available to the LATS tree."""

    @classmethod
    def get_all_strategies(cls) -> List[CleaningStrategy]:
        return [
            CleaningStrategy(
                name="iqr_standard",
                description="Standard IQR outlier filtering (1.5x IQR threshold)",
                func=clean_iqr_standard,
            ),
            CleaningStrategy(
                name="iqr_conservative",
                description="Conservative IQR filtering (3.0x IQR, only extreme outliers removed)",
                func=clean_iqr_conservative,
            ),
            CleaningStrategy(
                name="no_outlier_removal",
                description="Preserve full sample distribution without outlier filtering",
                func=clean_no_outlier_removal,
            ),
            CleaningStrategy(
                name="winsorization",
                description="Quantile Winsorization (clipping at 2nd and 98th percentiles, 100% sample retention)",
                func=clean_winsorization,
            ),
            CleaningStrategy(
                name="zscore_filter",
                description="Gaussian Z-score filtering (|z| <= 3.0 standard deviations)",
                func=clean_zscore,
            ),
            CleaningStrategy(
                name="median_imputation",
                description="Covariate median/mode imputation to maximize statistical power and sample retention",
                func=clean_median_imputation,
            ),
        ]

    @classmethod
    def get_strategy_names(cls) -> List[str]:
        return [s.name for s in cls.get_all_strategies()]

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        return cls.get_strategy_names()

    @classmethod
    def get_strategy(cls, name: str) -> Optional[CleaningStrategy]:
        for s in cls.get_all_strategies():
            if s.name == name:
                return s
        return None

    def apply_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        dependent_variable: str,
        independent_variables: List[str],
        covariates: Optional[List[str]] = None,
    ) -> CleaningResult:
        strat = self.get_strategy(strategy_name)
        if strat is None:
            raise KeyError(f"Cleaning strategy '{strategy_name}' not found in CleanerPool.")
        return strat.clean(df, dependent_variable, independent_variables, covariates)

