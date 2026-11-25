"""
Template Downloader for academic LaTeX journal templates.

Enables downloading and caching official journal templates from remote repositories
(CTAN, GitHub, Overleaf, conferences) as well as arbitrary custom template URLs.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import stat
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Official and trusted template repository endpoints
FAMOUS_JOURNAL_CATALOG: dict[str, dict[str, Any]] = {
    "ieee": {
        "name": "IEEE Conference & Transactions",
        "publisher": "IEEE",
        "description": "Standard IEEE two-column article/conference template (IEEEtran).",
        "url": "https://raw.githubusercontent.com/latextemplates/IEEEtran/master/IEEEtran.cls",
        "archive_url": "https://github.com/latextemplates/IEEEtran/archive/refs/heads/master.zip",
        "document_class": "IEEEtran",
    },
    "elsevier": {
        "name": "Elsevier Journal (elsarticle)",
        "publisher": "Elsevier",
        "description": "Elsevier flagship elsarticle manuscript template.",
        "url": "https://raw.githubusercontent.com/elsevier/elsarticle/master/elsarticle.cls",
        "archive_url": "https://github.com/elsevier/elsarticle/archive/refs/heads/master.zip",
        "document_class": "elsarticle",
    },
    "nature": {
        "name": "Springer Nature (sn-jnl)",
        "publisher": "Springer Nature",
        "description": "Springer Nature sn-jnl author template.",
        "url": "https://raw.githubusercontent.com/SpringerNature/sn-jnl/master/sn-jnl.cls",
        "archive_url": "https://github.com/SpringerNature/sn-jnl/archive/refs/heads/master.zip",
        "document_class": "sn-jnl",
    },
    "acm": {
        "name": "ACM Proceedings & Journals (acmart)",
        "publisher": "ACM",
        "description": "Association for Computing Machinery master acmart template.",
        "url": "https://raw.githubusercontent.com/borisveytsman/acmart/master/acmart.cls",
        "archive_url": "https://github.com/borisveytsman/acmart/archive/refs/heads/master.zip",
        "document_class": "acmart",
    },
    "plos": {
        "name": "PLOS ONE",
        "publisher": "PLOS",
        "description": "Public Library of Science open access article template.",
        "url": "https://raw.githubusercontent.com/PLOS/latex-templates/master/plos-template.tex",
        "archive_url": "https://github.com/PLOS/latex-templates/archive/refs/heads/master.zip",
        "document_class": "article",
    },
    "neurips": {
        "name": "NeurIPS Conference Template",
        "publisher": "NeurIPS / Curran Associates",
        "description": "Neural Information Processing Systems conference format.",
        "url": "https://raw.githubusercontent.com/neurips/neurips_style/master/neurips_2023.sty",
        "archive_url": "https://github.com/neurips/neurips_style/archive/refs/heads/master.zip",
        "document_class": "article",
    },
}


class TemplateDownloader:
    """Manages downloading, extracting, and caching remote LaTeX templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_catalog() -> dict[str, dict[str, Any]]:
        """Return the catalog of famous journal templates."""
        return FAMOUS_JOURNAL_CATALOG

    def download_template(
        self,
        template_name_or_url: str,
        custom_name: Optional[str] = None,
        timeout: int = 20,
    ) -> dict[str, Any]:
        """
        Download a template either by catalog identifier ('ieee', 'elsevier', etc.)
        or from a direct HTTP(S) URL pointing to a .zip, .tar.gz, or .tex file.
        """
        is_url = template_name_or_url.startswith("http://") or template_name_or_url.startswith("https://")

        if is_url:
            download_url = template_name_or_url
            template_id = custom_name or Path(download_url.split("?")[0]).stem.lower()
            metadata = {
                "id": template_id,
                "name": template_id.replace("_", " ").title(),
                "publisher": "Custom / External",
                "description": f"Downloaded from {download_url}",
            }
        else:
            template_id = template_name_or_url.lower()
            if template_id not in FAMOUS_JOURNAL_CATALOG:
                raise ValueError(
                    f"Unknown catalog template '{template_id}'. "
                    f"Available options: {list(FAMOUS_JOURNAL_CATALOG.keys())}"
                )
            catalog_entry = FAMOUS_JOURNAL_CATALOG[template_id]
            download_url = catalog_entry.get("archive_url") or catalog_entry.get("url")
            metadata = {
                "id": template_id,
                "name": catalog_entry["name"],
                "publisher": catalog_entry["publisher"],
                "description": catalog_entry["description"],
                "document_class": catalog_entry.get("document_class", "article"),
            }

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}", template_id):
            raise ValueError("Template identifier must be a simple directory name")
        target_folder = self.templates_dir / template_id
        target_folder.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading template '%s' from %s", template_id, download_url)

        headers = {"User-Agent": "ResearchAgent/1.0 (Academic Paper Generator)"}
        req = urllib.request.Request(download_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read(25 * 1024 * 1024 + 1)
                if len(content) > 25 * 1024 * 1024:
                    raise ValueError("Template download exceeds 25 MB")

            # Check if archive (zip or tar)
            if download_url.endswith(".zip") or (len(content) > 4 and content[:4] == b"PK\x03\x04"):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    members = z.infolist()
                    if len(members) > 2000 or sum(m.file_size for m in members) > 100 * 1024 * 1024:
                        raise ValueError("Template archive is too large")
                    for member in members:
                        self._validate_member(target_folder, member.filename)
                        if stat.S_ISLNK(member.external_attr >> 16):
                            raise ValueError("Template links are not allowed")
                    z.extractall(target_folder)
            elif download_url.endswith((".tar.gz", ".tgz")):
                with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as t:
                    members = t.getmembers()
                    if len(members) > 2000 or sum(m.size for m in members) > 100 * 1024 * 1024:
                        raise ValueError("Template archive is too large")
                    for member in members:
                        self._validate_member(target_folder, member.name)
                        if not (member.isfile() or member.isdir()):
                            raise ValueError("Template links and special files are not allowed")
                    t.extractall(target_folder, members=members)
            else:
                # Direct file download (.tex or .cls or .sty)
                filename = Path(download_url.split("?")[0]).name
                if not filename:
                    filename = "template.tex"
                self._validate_member(target_folder, filename)
                dest_file = target_folder / filename
                with open(dest_file, "wb") as f:
                    f.write(content)

            # Ensure template.json exists
            json_file = target_folder / "template.json"
            if not json_file.exists():
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

            return {
                "success": True,
                "template_id": template_id,
                "path": str(target_folder),
                "metadata": metadata,
            }

        except Exception as exc:
            logger.warning(
                "Could not download template '%s' from %s: %s. "
                "Checking if a pre-bundled local template exists.",
                template_id,
                download_url,
                exc,
            )
            # If local template already exists, succeed gracefully
            local_tex = target_folder / "template.tex"
            if local_tex.exists():
                return {
                    "success": True,
                    "template_id": template_id,
                    "path": str(target_folder),
                    "metadata": metadata,
                    "note": f"Using pre-bundled template ({exc})",
                }
            return {
                "success": False,
                "template_id": template_id,
                "error": str(exc),
            }

    @staticmethod
    def _validate_member(root: Path, name: str):
        # Check both path separator styles regardless of host operating system.
        normalized = name.replace("\\", "/")
        if ":" in normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("Unsafe template archive path")
        target = (root / normalized).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Template path escapes its destination")

    def download_all_famous_templates(self) -> dict[str, Any]:
        """Download or verify all famous journal templates in the catalog."""
        results: dict[str, Any] = {}
        for key in FAMOUS_JOURNAL_CATALOG:
            res = self.download_template(key)
            results[key] = res
        return results
