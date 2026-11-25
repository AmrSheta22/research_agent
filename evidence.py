"""Structured scholarly identity and explicitly reviewed source evidence."""
import html
import re
from difflib import SequenceMatcher


def clean_text(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html.unescape(str(value or "")))).strip()


def normalize_doi(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(value or "").strip(), flags=re.I).lower().rstrip(".")


def title_matches(left, right):
    key = lambda x: re.sub(r"[^\w]", "", clean_text(x).casefold())
    return SequenceMatcher(None, key(left), key(right)).ratio() >= .92


def canonical_record(item):
    title = clean_text((item.get("title") or [""])[0])
    authors = [{k:str(a[k]) for k in ("family","given","ORCID") if a.get(k)}
               for a in item.get("author", []) if a.get("family")]
    date = next((item[k]["date-parts"][0][0] for k in ("published-print", "published-online", "issued")
                 if item.get(k, {}).get("date-parts", [[]])[0]), None)
    doi = normalize_doi(item.get("DOI"))
    if not title or not authors or not date or not doi:
        return None
    names = [a["family"] + (", " + a["given"] if a.get("given") else "") for a in authors]
    venue = clean_text((item.get("container-title") or [""])[0])
    return dict(title=title, authors=authors, year=str(date), doi=doi, journal=venue,
                work_type=item.get("type", ""), url="https://doi.org/" + doi,
                citation=f"{'; '.join(names)} ({date}). {title}. {venue}. https://doi.org/{doi}",
                abstract=clean_text(item.get("abstract")), metadata_verified=True)


def source_text(record):
    """Retrieve complete abstracts from registry, OpenAlex, or publisher metadata."""
    abstract = clean_text(record.get("abstract"))
    url = record.get("url", "")
    if abstract:
        return {"text":abstract,"scope":"registry_abstract","url":url}
    import requests
    doi = normalize_doi(record.get("doi"))
    if not re.fullmatch(r"10\.\d{4,9}/\S+",doi):
        return {"text":"","scope":"unavailable","url":url}
    try:
        response=requests.get("https://api.openalex.org/works/https://doi.org/"+doi,timeout=15)
        if response.ok:
            work=response.json()
            # Check DOI identity before using a third-party abstract.
            if normalize_doi(work.get("doi"))==doi:
                index=work.get("abstract_inverted_index") or {}
                words={pos:word for word,positions in index.items() for pos in positions}
                text=clean_text(" ".join(words[i] for i in sorted(words)))
                if len(text)>100:
                    return {"text":text,"scope":"openalex_abstract","url":response.url}
    except (requests.RequestException,ValueError):
        pass
    try:
        from bs4 import BeautifulSoup
        response=requests.get("https://doi.org/"+doi,timeout=15,headers={"User-Agent":"ResearchEvidenceReview/1.0"})
        if response.ok:
            soup=BeautifulSoup(response.text,"html.parser")
            for node in soup.select('meta[name="citation_abstract"], meta[name="dc.description"], meta[name="DC.Description"], section.abstract, div.abstract, #abstract'):
                text=clean_text(node.get("content") or node.get_text(" ",strip=True))
                if len(text)>100:
                    return {"text":text,"scope":"publisher_abstract","url":response.url}
    except (requests.RequestException,ImportError):
        pass
    return {"text":"","scope":"unavailable","url":url}


def identity_issues(items):
    issues, seen = [], set()
    for obj in items:
        d = obj.model_dump() if hasattr(obj, "model_dump") else obj
        doi = normalize_doi(d.get("doi"))
        if not doi or not d.get("metadata_verified") or not d.get("evidence_quotes") or not d.get("reviewed_source"):
            issues.append(f"Source lacks verified identity or reviewed evidence: {d.get('title')}")
        if doi in seen:
            issues.append(f"Duplicate DOI: {doi}")
        seen.add(doi)
        cited_dois={normalize_doi(value) for value in re.findall(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s]+)",d.get("citation", ""),flags=re.I)}
        if doi and doi not in cited_dois:
            issues.append(f"Citation DOI differs from source identity: {d.get('title')}")
    return issues


def verify_registry_identity(record):
    """Resolve the exact DOI again; caller-supplied verification flags are insufficient."""
    doi=normalize_doi(record.get("doi"))
    if not doi:
        return ["Missing DOI identity"]
    try:
        from habanero import Crossref
        response=Crossref().works(ids=doi)
        resolved=canonical_record(response.get("message",{}))
        if not resolved or resolved["doi"]!=doi:
            return [f"Registry did not resolve the claimed DOI: {doi}"]
        problems=[]
        if not title_matches(record.get("title",""),resolved["title"]):
            problems.append("title")
        if str(record.get("year"))!=resolved["year"]:
            problems.append("year")
        families=lambda authors:[clean_text(a.get("family","")).casefold() for a in authors]
        if families(record.get("authors",[]))!=families(resolved["authors"]):
            problems.append("authors")
        if clean_text(record.get("journal","")).casefold()!=resolved["journal"].casefold():
            problems.append("venue")
        return [f"DOI {doi} identity differs in {', '.join(problems)}"] if problems else []
    except Exception as exc:
        return [f"DOI {doi} identity verification incomplete: {exc}"]
