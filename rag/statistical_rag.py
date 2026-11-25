"""
Retrieval-Augmented Generation (RAG) engine for statistical test textbooks.
Indexes open-access statistical textbooks on hypothesis testing, ANOVA, linear mixed models,
pseudo-replication, variance partitioning, power analysis, and equivalence testing.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from research_agent.rag.downloader import STATISTICAL_BOOKS_DIR, download_or_build_textbooks

logger = logging.getLogger(__name__)


class TextbookChunk:
    """Represents a chunk of statistical textbook text with metadata."""

    def __init__(self, book_title: str, chapter_title: str, text: str, chunk_id: str):
        self.book_title = book_title
        self.chapter_title = chapter_title
        self.text = text.strip()
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return f"<TextbookChunk {self.chunk_id}: {self.chapter_title}>"


class StatisticalRAG:
    """
    Search and retrieval engine over the statistical textbook knowledge base.
    Uses BM25-style lexical scoring with term frequency, inverse document frequency,
    and chapter heading boosting to retrieve high-precision guidance on statistical tests.
    """

    _instance: Optional[StatisticalRAG] = None

    def __init__(self, books_dir: Optional[Path] = None):
        self.books_dir = Path(books_dir) if books_dir else STATISTICAL_BOOKS_DIR
        self.chunks: List[TextbookChunk] = []
        self.chunk_tokens: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self._load_and_index()

    @classmethod
    def get_instance(cls, books_dir: Optional[Path] = None) -> StatisticalRAG:
        """Singleton accessor for the Statistical RAG engine."""
        if cls._instance is None:
            cls._instance = cls(books_dir=books_dir)
        return cls._instance

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust lowercased tokenization for information retrieval."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\^]+\b", text.lower())
        return [w for w in words if len(w) > 1]

    def _load_and_index(self):
        """Load markdown textbook volumes and chunk by sections."""
        if self.books_dir == STATISTICAL_BOOKS_DIR:
            download_or_build_textbooks()
        if not self.books_dir.exists():
            return

        book_files = sorted(list(self.books_dir.glob("*.md")))
        chunk_count = 0

        for book_file in book_files:
            try:
                content = book_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Could not read textbook file %s: %s", book_file, e)
                continue

            lines = content.splitlines()
            current_book = book_file.stem.replace("_", " ").title()
            current_chapter = "Overview"
            current_paragraphs: List[str] = []

            for line in lines:
                if line.startswith("# "):
                    current_book = line.lstrip("# ").strip()
                elif line.startswith("## "):
                    if current_paragraphs:
                        text = "\n".join(current_paragraphs).strip()
                        if len(text.split()) >= 30:
                            chunk_id = f"{book_file.stem}_{chunk_count}"
                            self.chunks.append(
                                TextbookChunk(
                                    book_title=current_book,
                                    chapter_title=current_chapter,
                                    text=text,
                                    chunk_id=chunk_id,
                                )
                            )
                            chunk_count += 1
                        current_paragraphs = []
                    current_chapter = line.lstrip("# ").strip()
                else:
                    current_paragraphs.append(line)

            if current_paragraphs:
                text = "\n".join(current_paragraphs).strip()
                if len(text.split()) >= 30:
                    chunk_id = f"{book_file.stem}_{chunk_count}"
                    self.chunks.append(
                        TextbookChunk(
                            book_title=current_book,
                            chapter_title=current_chapter,
                            text=text,
                            chunk_id=chunk_id,
                        )
                    )
                    chunk_count += 1

        # Build BM25 index
        n_docs = len(self.chunks)
        if n_docs == 0:
            logger.warning("No statistical textbook chunks were indexed.")
            return

        total_tokens = 0
        self.chunk_tokens = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.chapter_title + " " + chunk.text)
            self.chunk_tokens.append(tokens)
            total_tokens += len(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.avg_doc_len = total_tokens / n_docs
        logger.info("StatisticalRAG successfully indexed %d textbook chunks from %d books.", n_docs, len(book_files))

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the statistical textbook knowledge base using BM25 scoring.
        Returns top_k most relevant chunks with source metadata and similarity score.
        """
        if not self.chunks or not query_text.strip():
            return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return []

        k1 = 1.5
        b = 0.75
        n_docs = len(self.chunks)
        scores: List[Tuple[float, int]] = []

        for idx, doc_tokens in enumerate(self.chunk_tokens):
            score = 0.0
            doc_len = len(doc_tokens)
            doc_token_counts: Dict[str, int] = {}
            for t in doc_tokens:
                doc_token_counts[t] = doc_token_counts.get(t, 0) + 1

            for qt in q_tokens:
                if qt not in self.doc_freq:
                    continue
                df = self.doc_freq[qt]
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = doc_token_counts.get(qt, 0)
                term_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))))
                score += term_score

            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = scores[:top_k]

        results: List[Dict[str, Any]] = []
        for score, idx in top_results:
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": round(score, 4),
                    "book_title": chunk.book_title,
                    "chapter_title": chunk.chapter_title,
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                }
            )

        return results

    def get_context_for_prompt(self, query_text: str, top_k: int = 2) -> str:
        """Format retrieved textbook snippets as an injection-ready prompt block."""
        matches = self.query(query_text, top_k=top_k)
        if not matches:
            return ""

        blocks = []
        for m in matches:
            blocks.append(
                f"### [Textbook Reference: {m['book_title']} - {m['chapter_title']}]\n{m['text']}"
            )
        return "\n\n".join(blocks)
