"""
Persistent memory store for continuous learning across research sessions.

Records learned variable mappings, test results, prompt refinements,
and user feedback so the agent improves over time rather than starting
from an isolated context every time.
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from contextlib import contextmanager
from filelock import FileLock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Persistent memory engine.

    Categories of learned knowledge:
      • variable_types    – historical mappings of column names to statistical types
      • test_outcomes     – which statistical tests were selected and whether they were significant
      • confounders       – commonly identified confounding pairs
      • prompt_learnings  – effective prompts and refinements
      • user_corrections  – explicit user overrides and corrections
    """

    def __init__(self, backend: str = "json", path: str = ".research_agent/memory"):
        if backend != "json":
            raise ValueError("Only the json memory backend is implemented")
        self._lock = threading.RLock()
        self._file_lock = FileLock(os.path.join(path, "learnings.lock"))
        self.backend = backend
        self.path = path
        self._data: dict[str, dict[str, Any]] = {
            "variable_types": {},
            "test_outcomes": {},
            "confounders": {},
            "prompt_learnings": {},
            "user_corrections": {},
        }
        self._load()

    def _get_file_path(self) -> str:
        return os.path.join(self.path, "learnings.json")

    def _load(self):
        """Load persistent memories from disk."""
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self._data = {k: v for k, v in loaded.items() if isinstance(v, dict)}
                logger.info("Loaded persistent memories from %s", file_path)
            except Exception as exc:
                logger.warning("Failed to load memory from %s: %s", file_path, exc)

    MAX_ENTRIES_PER_CATEGORY = 500

    def _save(self):
        """Flush memories to disk atomically."""
        try:
            os.makedirs(self.path, exist_ok=True)
            file_path = self._get_file_path()
            tmp_path = file_path + f".{os.getpid()}_{time.time_ns()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            # On Windows, brief retry loop if destination is momentarily locked
            for attempt in range(5):
                try:
                    os.replace(tmp_path, file_path)
                    break
                except OSError:
                    time.sleep(0.02)
            else:
                try:
                    os.replace(tmp_path, file_path)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Failed to save memory to %s: %s", self.path, exc)

    # ------------------------------------------------------------------
    # Generic CRUD
    # ------------------------------------------------------------------
    @contextmanager
    def _transaction(self):
        os.makedirs(self.path, exist_ok=True)
        with self._lock, self._file_lock:
            self._load()
            yield

    def save_learning(self, category: str, key: str, value: Any):
        """Merge a single update under a cross-process lock before atomic replacement."""
        with self._transaction():
            cat_dict = self._data.setdefault(category, {})
            if len(cat_dict) >= self.MAX_ENTRIES_PER_CATEGORY and key not in cat_dict:
                oldest = min(cat_dict, key=lambda k: cat_dict[k].get("updated_at", 0))
                cat_dict.pop(oldest, None)
            cat_dict[key] = {"value": value, "updated_at": time.time()}
            self._save()

    def get_learning(self, category: str, key: str) -> Optional[Any]:
        """Retrieve a specific learned item."""
        entry = self._data.get(category, {}).get(key)
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        return entry

    def get_all_learnings(self, category: str) -> dict[str, Any]:
        """Retrieve all entries in a category."""
        return {
            k: (v["value"] if isinstance(v, dict) and "value" in v else v)
            for k, v in self._data.get(category, {}).items()
        }

    # ------------------------------------------------------------------
    # Domain-specific continuous learning methods
    # ------------------------------------------------------------------
    def record_variable_type_pattern(self, column_name: str, detected_type: str, was_correct: bool = True):
        """Learn and refine column name to data type mappings."""
        with self._transaction():
            norm_col = column_name.strip().lower()
            stats = self.get_learning("variable_types", norm_col) or {}
            counts = dict(stats.get("type_counts", {})) if isinstance(stats, dict) else {}
            delta = 1 if was_correct else -1
            counts[detected_type] = max(0, counts.get(detected_type, 0) + delta)
            # Avoid crash on empty/zero counts
            if any(v > 0 for v in counts.values()):
                preferred = max(counts, key=counts.get)
            else:
                preferred = detected_type
            self.save_learning("variable_types", norm_col, {
                "type_counts": counts,
                "preferred_type": preferred,
            })

    def get_learned_variable_type(self, column_name: str) -> Optional[str]:
        """Check if we have high-confidence prior knowledge about this column's type."""
        norm_col = column_name.strip().lower()
        stats = self.get_learning("variable_types", norm_col)
        if isinstance(stats, dict) and "preferred_type" in stats:
            return stats["preferred_type"]
        return None

    def record_test_outcome(
        self,
        design: str,
        iv_type: str,
        dv_type: str,
        test_used: str,
        significant: bool,
    ):
        """Record historical test execution outcomes."""
        with self._transaction():
            key = f"{design}::{iv_type}::{dv_type}"
            history = self.get_learning("test_outcomes", key) or []
            if not isinstance(history, list):
                history = []
            history.append({
                "test": test_used,
                "significant": significant,
                "timestamp": time.time(),
            })
            self.save_learning("test_outcomes", key, history[-20:])  # keep last 20

    def record_successful_prompt(self, skill_name: str, context_summary: str, result_quality: float = 1.0):
        """Record a successful generation pattern."""
        self.save_learning("prompt_learnings", f"{skill_name}::{time.strftime('%Y%m')}", {
            "last_success": context_summary,
            "quality": result_quality,
        })

    def record_user_correction(self, field: str, original_value: Any, corrected_value: Any):
        """Learn when a user overrides the agent's prediction."""
        key = f"{field}::{str(original_value)[:50]}"
        self.save_learning("user_corrections", key, {
            "original": original_value,
            "corrected": corrected_value,
            "timestamp": time.time(),
        })
        logger.info("Recorded continuous learning user correction for %s: %s -> %s", field, original_value, corrected_value)

    def get_context_enhancements(self, context_summary: str) -> list[str]:
        """Surface relevant past learnings to augment the prompt context."""
        enhancements: list[str] = []
        corrections = self.get_all_learnings("user_corrections")
        for k, v in list(corrections.items())[-5:]:
            if isinstance(v, dict):
                enhancements.append(f"Past user correction for {k}: preferred {v.get('corrected')}")
        return enhancements
