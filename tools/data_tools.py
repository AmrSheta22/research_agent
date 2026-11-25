"""
Data-processing tools: file reading, cleaning, type detection, and variable summaries.

All logic is self-contained — nothing is imported from the legacy ``llm_generation`` package.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional, Type

import chardet
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ============================================================================
# Internal helpers (ported from utils_clean_typcast / utils_helper)
# ============================================================================

_NA_STRINGS = {"none", "null", "unknown", "nan"}


def _is_na_string(x: Any) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s == "" or s in _NA_STRINGS


def clean_na(series: pd.Series) -> pd.Series:
    """Drop NA-like values (NaN, None, 'null', 'unknown', whitespace-only)."""
    if series.empty:
        return series
    series_clean = series.copy()
    mask = series_clean.apply(_is_na_string)
    return series_clean[~mask]


def _clean_numeric_str(x: Any) -> Any:
    if isinstance(x, (int, float, np.number)):
        return x
    s = str(x).strip()
    # If date/time pattern (e.g. 2026-05-23 or 14:30:00), preserve as non-numeric
    if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{1,2}", s):
        return s
    s = s.replace(",", "")
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = s.replace("%", "")
    match = re.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", s)
    if match:
        return match.group(0)
    return s


def detect_numeric_col(series: pd.Series) -> bool:
    """Return ``True`` if ≥ 90 % of non-NA values can be coerced to a number."""
    cleaned_series = clean_na(series)
    if cleaned_series.empty:
        return False
    potential_numbers = cleaned_series.apply(_clean_numeric_str)
    numeric_converted = pd.to_numeric(potential_numbers, errors="coerce")
    valid_count = numeric_converted.notna().sum()
    total_count = len(cleaned_series)
    if total_count == 0:
        return False
    return (valid_count / total_count) > 0.9


def detect_categorical_col(series: pd.Series) -> tuple[bool, list]:
    """Return ``(True, categories)`` when uniqueness < 20 % and categories repeat > 3×."""
    cleaned_series = clean_na(series)
    if cleaned_series.empty:
        return False, []
    total_count = len(cleaned_series)
    if total_count == 0:
        return False, []
    val_counts = cleaned_series.value_counts()
    unique_count = len(val_counts)
    uniqueness_ratio = unique_count / total_count
    target_categories: list = []
    if uniqueness_ratio < 0.20:
        for val, count in val_counts.items():
            if count > 3:
                target_categories.append(val)
        if len(target_categories) > 0:
            return True, target_categories
    return False, []


def determine_type_of_variable(col_name: str, data: pd.DataFrame) -> Optional[str]:
    """Detect whether *col_name* is numeric / categorical / binary."""
    col_data = data[col_name]
    cleaned_col = clean_na(col_data)
    if cleaned_col.empty or cleaned_col.nunique() <= 1:
        return None  # Constant or degenerate column cannot be used

    cat_result = detect_categorical_col(col_data)
    is_cat = cat_result[0]
    cat_values = cat_result[1]
    is_num = detect_numeric_col(col_data)

    # 1. Exactly two distinct non-null values -> binary
    if cleaned_col.nunique() == 2 or (is_cat and len(cat_values) == 2):
        return "binary"

    # 2. Continuous numeric with sufficient variation
    if is_num and (not is_cat or len(cat_values) >= 5):
        return "numeric"

    # 3. Categorical factors (between 3 and 10 categories)
    if is_cat:
        if 2 < len(cat_values) <= 10:
            return "categorical"
        elif len(cat_values) > 10:
            return None
    elif is_num:
        return "numeric"

    return None


def ensure_correct_data_type(data_type: str, series: pd.Series) -> bool:
    """Check whether *series* can safely be cast to *data_type*."""
    try:
        if data_type == "numeric":
            series.astype(float)
            return True
        elif data_type in ("categorical", "binary"):
            series.astype("category")
            return True
    except (ValueError, TypeError):
        return False
    return False


def convert_to_numeric(series: pd.Series) -> pd.Series:
    """Clean and coerce a series to numeric values."""
    if series.empty:
        return series
    potential_numbers = series.apply(_clean_numeric_str)
    return pd.to_numeric(potential_numbers, errors="coerce")


def convert_to_categorical(series: pd.Series) -> pd.Series:
    """Clean missing values while preserving every observed category."""
    if series.empty:
        return series.astype("category")

    def _clean_val(x: Any) -> Any:
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s == "" or s in _NA_STRINGS:
            return np.nan
        return x

    cleaned_values = series.apply(_clean_val)
    return cleaned_values.astype("category")


def read_file_safely(
    file_path: str,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    max_file_size_mb: int = 100,
) -> Optional[pd.DataFrame]:
    """Read CSV / Excel / text files with auto-detected encoding, delimiter, header, and size guards."""
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return None

    # File size safety guard
    file_size_bytes = os.path.getsize(file_path)
    if file_size_bytes > (max_file_size_mb * 1024 * 1024):
        logger.error(
            "File size (%.2f MB) exceeds safety threshold of %d MB: %s",
            file_size_bytes / (1024 * 1024),
            max_file_size_mb,
            file_path,
        )
        return None

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(file_path, header=0)
            df = df.reset_index(drop=True)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as exc:
            logger.error("Error reading Excel file: %s", exc)
            return None

    # CSV / text files
    if encoding is None:
        try:
            with open(file_path, "rb") as fh:
                result = chardet.detect(fh.read(100_000))
                encoding = result.get("encoding") or "utf-8"
        except Exception:
            encoding = "utf-8"

    if delimiter is None:
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as fh:
                sample = fh.read(4096)
            delimiters = [",", ";", "|", "\t"]
            delimiter_counts = {d: sample.count(d) for d in delimiters}
            delimiter = max(delimiter_counts, key=delimiter_counts.get)
            if delimiter_counts.get(delimiter, 0) == 0:
                delimiter = ","
        except Exception:
            delimiter = ","

    try:
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
            skipinitialspace=True,
            header="infer",
        )
        df = df.reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                header="infer",
            )
            df = df.reset_index(drop=True)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as exc:
            logger.error("Error parsing CSV file: %s", exc)
            return None
    except Exception as exc:
        logger.error("Error reading CSV file: %s", exc)
        return None


def clean_dataframe(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: list[str],
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Clean *df*: NA removal, type-casting, IQR outlier removal for numerics."""
    if covariates is None:
        covariates = []
    relevant_columns = [dependent_variable] + independent_variables + covariates

    # 1. Remove rows with invalid values across all relevant columns
    valid_indices = set(df.index)
    for col in relevant_columns:
        if col in df.columns:
            cleaned_col_series = clean_na(df[col])
            valid_indices = valid_indices.intersection(set(cleaned_col_series.index))

    df_clean = df.loc[list(valid_indices)].copy()
    if df_clean.empty:
        return df_clean

    # 2. Typecasting and outlier removal
    for col in relevant_columns:
        if col not in df_clean.columns:
            continue
        col_type = determine_type_of_variable(col, df_clean)

        if col_type == "numeric":
            df_clean[col] = convert_to_numeric(df_clean[col])
            df_clean[col + "_original"] = df_clean[col]
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]
        elif col_type in ("categorical", "binary"):
            df_clean[col] = convert_to_categorical(df_clean[col])
            df_clean[col + "_original"] = df_clean[col]

    # Drop rows that became NaN due to categorical filtering or conversion
    df_clean = df_clean.dropna(subset=[c for c in relevant_columns if c in df_clean.columns])
    df_clean = df_clean.dropna(subset=[col + "_original" for col in relevant_columns if col + "_original" in df_clean.columns])
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def produce_variable_summary(column_name: str, column_type: str, df: pd.DataFrame) -> str:
    """Produce a human-readable summary for a variable."""
    if column_type == "numeric":
        return df[column_name].describe().to_string()
    elif column_type in ("categorical", "binary"):
        return df[column_name].value_counts().to_string()
    return ""


# ============================================================================
# Tool classes
# ============================================================================


class _ReadFileInput(BaseModel):
    file_path: str = Field(description="Path to the CSV / Excel / text file to read.")


class ReadFileTool(BaseTool):
    """Read a tabular file (CSV, Excel, .txt, .dat) into a DataFrame."""

    name = "read_file"
    description = (
        "Reads a CSV, Excel, or text-based data file with automatic header, "
        "delimiter, and encoding detection. Returns the loaded DataFrame."
    )
    input_schema: type[BaseModel] = _ReadFileInput

    def execute(self, **kwargs: Any) -> ToolResult:
        file_path: str = kwargs["file_path"]
        if not os.path.isfile(file_path):
            return ToolResult.fail(f"File not found: {file_path}")
        df = read_file_safely(file_path)
        if df is None:
            return ToolResult.fail(f"Failed to read file: {file_path}")
        return ToolResult.ok(df)


class _CleanDataFrameInput(BaseModel):
    dependent_var: str = Field(description="Name of the dependent variable column.")
    independent_vars: List[str] = Field(description="List of independent variable column names.")
    covariates: List[str] = Field(default_factory=list, description="List of covariate column names.")


class CleanDataFrameTool(BaseTool):
    """Clean a DataFrame: NA handling, type-casting, and IQR outlier removal."""

    name = "clean_dataframe"
    description = (
        "Cleans a DataFrame by removing NA-like values, casting columns to "
        "appropriate types, and removing IQR outliers for numeric columns."
    )
    input_schema: type[BaseModel] = _CleanDataFrameInput

    def execute(self, **kwargs: Any) -> ToolResult:
        df: pd.DataFrame = kwargs["df"]
        dependent_var: str = kwargs.get("dependent_var") or kwargs.get("dependent_variable", "")
        independent_vars: list[str] = kwargs.get("independent_vars") or kwargs.get("independent_variables", [])
        covariates: list[str] = kwargs.get("covariates", [])

        if not isinstance(df, pd.DataFrame):
            return ToolResult.fail("Expected a pandas DataFrame for 'df'.")

        cleaned = clean_dataframe(df, dependent_var, independent_vars, covariates)
        return ToolResult.ok(cleaned)


class _DetectVariableTypeInput(BaseModel):
    column_name: str = Field(description="Name of the column to inspect.")


class DetectVariableTypeTool(BaseTool):
    """Detect whether a column is numeric, categorical, or binary."""

    name = "detect_variable_type"
    description = (
        "Detects the statistical type of a DataFrame column. "
        "Returns one of: 'numeric', 'categorical', 'binary', or None."
    )
    input_schema: type[BaseModel] = _DetectVariableTypeInput

    def execute(self, **kwargs: Any) -> ToolResult:
        column_name: str = kwargs.get("column_name")
        df: Optional[pd.DataFrame] = kwargs.get("df") if "df" in kwargs else kwargs.get("data")
        if df is None:
            return ToolResult.fail("No DataFrame provided.")
        if not column_name or column_name not in df.columns:
            return ToolResult.fail(f"Column '{column_name}' not found in DataFrame.")
        var_type = determine_type_of_variable(column_name, df)
        return ToolResult.ok(var_type)


class _ProduceVariableSummaryInput(BaseModel):
    column_name: str = Field(description="Name of the column to summarise.")
    column_type: str = Field(
        description="Type of the column: 'numeric', 'categorical', or 'binary'."
    )


class ProduceVariableSummaryTool(BaseTool):
    """Produce a descriptive summary string for a variable column."""

    name = "produce_variable_summary"
    description = (
        "Returns descriptive statistics (for numeric) or value counts "
        "(for categorical / binary) as a human-readable string."
    )
    input_schema: type[BaseModel] = _ProduceVariableSummaryInput

    def execute(self, **kwargs: Any) -> ToolResult:
        column_name: str = kwargs.get("column_name")
        column_type: str = kwargs.get("column_type")
        df: Optional[pd.DataFrame] = kwargs.get("df") if "df" in kwargs else kwargs.get("data")
        if df is None:
            return ToolResult.fail("No DataFrame provided.")
        if not column_name or column_name not in df.columns:
            return ToolResult.fail(f"Column '{column_name}' not found in DataFrame.")
        summary = produce_variable_summary(column_name, column_type, df)
        return ToolResult.ok(summary)


def get_file_header_and_first_five_rows(file_path: str) -> str:
    """Reads the header and first 5 rows of a dataset."""
    df = read_file_safely(file_path)
    if df is None:
        return "Error reading file."
    return df.head(5).to_string()


class _DatasetQueryInput(BaseModel):
    query: str = Field(description="Description of the statistic, calculation, or question about the data.")
    code: Optional[str] = Field(default=None, description="Optional read-only summary expression over df; arbitrary Python is not supported.")


class DatasetQueryTool(BaseTool):
    """
    Executes Python data calculations directly against the working DataFrame.
    Enables agents, critics, and revisers to verify numbers, compute subgroup statistics,
    and ground empirical claims in actual dataset values rather than guessing.
    """

    name = "dataset_query"
    description = (
        "Queries and computes statistics directly on the dataset. Accepts a Python expression "
        "or code snippet to evaluate against `df` (e.g. `df.groupby('group_var')['outcome_var'].describe()`), "
        "or a query description. Returns exact numerical outputs."
    )
    input_schema: type[BaseModel] = _DatasetQueryInput

    def execute(self, **kwargs: Any) -> ToolResult:
        df: Optional[pd.DataFrame] = kwargs.get("df") if "df" in kwargs else kwargs.get("data")
        query: str = kwargs.get("query", "")
        code: Optional[str] = kwargs.get("code")

        if df is None:
            return ToolResult.fail("No DataFrame provided for dataset query.")

        if code and code.strip():
            try:
                from research_agent.tools.safe_query import evaluate_summary
                result = evaluate_summary(code, df)
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    return ToolResult.ok(result.to_string())
                return ToolResult.ok(str(result))
            except Exception as exc:
                return ToolResult.fail(f"Unsupported or invalid summary expression: {exc}")

        # If no code, perform general column inspect or overview based on query
        matching_cols = [c for c in df.columns if c.lower() in query.lower()]
        if matching_cols:
            res_dict = {}
            for c in matching_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    res_dict[c] = df[c].describe().to_dict()
                else:
                    res_dict[c] = df[c].value_counts().to_dict()
            return ToolResult.ok(str(res_dict))

        return ToolResult.ok(f"Dataset shape: {df.shape}. Columns: {list(df.columns)}")

