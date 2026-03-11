"""
HuggingFace Paper Collector

Automatically discovers new and similar papers via Semantic Scholar
and adds them to a HuggingFace collection.
"""

import os
import time
import yaml
import logging
import argparse
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from huggingface_hub import HfApi, login

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REC_BASE = "https://api.semanticscholar.org/recommendations/v1/papers"
S2_FIELDS = "externalIds,title,abstract,year,citationCount,publicationDate,openAccessPdf,fieldsOfStudy"

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests (respect S2 limits)
MAX_RETRIES = 3


class SemanticScholarClient:
    """Thin wrapper around the Semantic Scholar API."""

    def __init__(self, api_key: str | None = None):
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _get(self, url: str, params: dict | None = None) -> dict | None:
        for attempt in range(MAX_RETRIES):
            time.sleep(REQUEST_DELAY)
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = REQUEST_DELAY * (2 ** attempt)
                    log.warning(f"Rate limited (429), retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"S2 API error: {e}, retrying...")
                    continue
                log.warning(f"S2 API error after {MAX_RETRIES} attempts: {e}")
                return None
        return None

    def search(self, query: str, limit: int = 20, year: str | None = None) -> list[dict]:
        """Search for papers by keyword query."""
        params = {"query": query, "limit": min(limit, 100), "fields": S2_FIELDS}
        if year:
            params["year"] = year
        data = self._get(f"{S2_BASE}/paper/search", params)
        if data and "data" in data:
            return data["data"]
        return []

    def get_recommendations(self, paper_ids: list[str], limit: int = 20) -> list[dict]:
        """Get recommended papers based on seed paper IDs."""
        payload = {"positivePaperIds": paper_ids}
        params = {"limit": min(limit, 100), "fields": S2_FIELDS}
        for attempt in range(MAX_RETRIES):
            time.sleep(REQUEST_DELAY)
            try:
                resp = self.session.post(
                    S2_REC_BASE, json=payload, params=params, timeout=30
                )
                if resp.status_code == 429:
                    wait = REQUEST_DELAY * (2 ** attempt)
                    log.warning(f"Rate limited (429), retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data.get("recommendedPapers", [])
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"S2 recommendations error: {e}, retrying...")
                    continue
                log.warning(f"S2 recommendations error after {MAX_RETRIES} attempts: {e}")
                return []
        return []

    def get_citations(self, paper_id: str, limit: int = 100) -> list[dict]:
        """Get papers that cite a given paper."""
        all_citations = []
        offset = 0
        while offset < limit:
            batch = min(100, limit - offset)
            params = {"fields": f"citingPaper.{S2_FIELDS}", "limit": batch, "offset": offset}
            data = self._get(f"{S2_BASE}/paper/{paper_id}/citations", params)
            if not data or "data" not in data:
                break
            batch_data = data["data"]
            if not batch_data:
                break
            for entry in batch_data:
                citing = entry.get("citingPaper")
                if citing and citing.get("title"):
                    all_citations.append(citing)
            offset += len(batch_data)
            if len(batch_data) < batch:
                break  # No more pages
        return all_citations

    def get_paper(self, paper_id: str) -> dict | None:
        """Fetch a single paper by ID."""
        params = {"fields": S2_FIELDS}
        return self._get(f"{S2_BASE}/paper/{paper_id}", params)


def load_config(path: str) -> dict:
    """Load and resolve config, substituting env vars."""
    with open(path) as f:
        raw = f.read()

    # Substitute ${ENV_VAR} patterns with environment variables
    import re
    def replace_env(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    resolved = re.sub(r"\$\{(\w+)\}", replace_env, raw)
    return yaml.safe_load(resolved)


def extract_arxiv_id(paper: dict) -> str | None:
    """Extract arXiv ID from a Semantic Scholar paper object."""
    ext = paper.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    if arxiv_id:
        return arxiv_id
    return None


def passes_filters(paper: dict, filters: dict, cutoff_date: datetime | None) -> bool:
    """Check if a paper passes the configured filters."""
    # Check publication date
    if cutoff_date and paper.get("publicationDate"):
        try:
            pub_date = datetime.strptime(paper["publicationDate"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            if pub_date < cutoff_date:
                return False
        except ValueError:
            pass  # If date parsing fails, don't filter on date

    # Check citation count
    min_cites = filters.get("min_citations", 0)
    if (paper.get("citationCount") or 0) < min_cites:
        return False

    # Check open access
    if filters.get("require_open_access") and not paper.get("openAccessPdf"):
        return False

    # Check fields of study
    allowed_fields = filters.get("fields_of_study", [])
    if allowed_fields:
        raw_fields = paper.get("fieldsOfStudy") or []
        paper_fields = set()
        for f in raw_fields:
            if isinstance(f, str):
                paper_fields.add(f)
            elif isinstance(f, dict):
                paper_fields.add(f.get("category", ""))
        if not paper_fields.intersection(set(allowed_fields)):
            return False

    return True


def discover_papers(s2: SemanticScholarClient, config: dict) -> dict[str, dict]:
    """
    Run all discovery methods and return deduplicated papers.
    Returns dict of arxiv_id -> {paper_data, source_note}.
    """
    filters = config.get("filters", {})
    max_age = filters.get("max_age_days", 30)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age) if max_age else None

    # Year filter for keyword search (approximate — S2 search supports year ranges)
    year_filter = None
    if cutoff:
        year_filter = f"{cutoff.year}-"

    discovered: dict[str, dict] = {}

    # --- Keyword-based discovery ---
    for search_cfg in config.get("keyword_searches", []):
        query = search_cfg["query"]
        max_results = search_cfg.get("max_results", 20)
        log.info(f"Searching for: '{query}' (max {max_results})")

        papers = s2.search(query, limit=max_results, year=year_filter)
        for p in papers:
            arxiv_id = extract_arxiv_id(p)
            if arxiv_id and passes_filters(p, filters, cutoff):
                if arxiv_id not in discovered:
                    discovered[arxiv_id] = {
                        "paper": p,
                        "note": f"keyword: {query}",
                    }
        log.info(f"  Found {len(papers)} results, {len(discovered)} total after filtering")

    # --- Citation tracking ---
    for tracked in config.get("track_citations", []):
        paper_id = tracked["id"]
        label = tracked.get("note", paper_id)
        max_cites = tracked.get("max_results", 200)
        log.info(f"Tracking citations of: {label} ({paper_id})")

        citations = s2.get_citations(paper_id, limit=max_cites)
        count_before = len(discovered)
        for p in citations:
            arxiv_id = extract_arxiv_id(p)
            # No date cutoff for citations — we want all citers
            if arxiv_id and passes_filters(p, filters, cutoff_date=None):
                if arxiv_id not in discovered:
                    discovered[arxiv_id] = {
                        "paper": p,
                        "note": f"cites: {label}",
                    }
        log.info(f"  Found {len(citations)} citing papers, {len(discovered) - count_before} new after filtering")

    # --- Similar-paper discovery ---
    seed_ids = [s["id"] for s in config.get("seed_papers", [])]
    seed_notes = {s["id"]: s.get("note", "") for s in config.get("seed_papers", [])}

    if seed_ids:
        log.info(f"Getting recommendations from {len(seed_ids)} seed papers")
        recs = s2.get_recommendations(seed_ids, limit=50)
        for p in recs:
            arxiv_id = extract_arxiv_id(p)
            if arxiv_id and passes_filters(p, filters, cutoff):
                if arxiv_id not in discovered:
                    discovered[arxiv_id] = {
                        "paper": p,
                        "note": "recommended (similar to seeds)",
                    }
        log.info(f"  Got {len(recs)} recommendations, {len(discovered)} total after filtering")

    return discovered


def add_to_collection(
    hf: HfApi, collection_slug: str, papers: dict[str, dict], dry_run: bool = False
) -> int:
    """Add discovered papers to the HuggingFace collection. Returns count of newly added."""
    added = 0

    for arxiv_id, info in papers.items():
        title = info["paper"].get("title", "untitled")
        note = info["note"]

        # HF paper IDs are just the arxiv ID (e.g. "2202.05262")
        paper_id = arxiv_id

        if dry_run:
            log.info(f"  [DRY RUN] Would add: {paper_id} — {title} ({note})")
            added += 1
            continue

        try:
            hf.add_collection_item(
                collection_slug=collection_slug,
                item_id=paper_id,
                item_type="paper",
                note=note[:500],  # HF note limit
                exists_ok=True,
            )
            log.info(f"  Added: {paper_id} — {title}")
            added += 1
        except Exception as e:
            log.warning(f"  Failed to add {paper_id}: {e}")

    return added


def main():
    parser = argparse.ArgumentParser(description="Collect papers into a HuggingFace collection")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Discover papers but don't add to collection"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        return
    config = load_config(str(config_path))

    # Init Semantic Scholar client
    s2_key = config.get("semantic_scholar", {}).get("api_key")
    if s2_key and s2_key.startswith("$"):
        s2_key = None  # Env var not resolved
    s2 = SemanticScholarClient(api_key=s2_key)

    # Init HuggingFace
    collection_slug = config["huggingface"]["collection_slug"]
    if collection_slug.startswith("$"):
        log.error("HF_COLLECTION_SLUG not set")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not args.dry_run:
        if not hf_token:
            log.error("HF_TOKEN not set — needed to modify collections")
            return
        login(token=hf_token)

    hf = HfApi()

    # Discover
    log.info("=== Starting paper discovery ===")
    papers = discover_papers(s2, config)
    log.info(f"Discovered {len(papers)} unique papers matching criteria")

    if not papers:
        log.info("No new papers found. Done.")
        return

    # Add to collection
    log.info(f"=== Adding papers to collection: {collection_slug} ===")
    added = add_to_collection(hf, collection_slug, papers, dry_run=args.dry_run)
    log.info(f"Done! Added {added} papers.")


if __name__ == "__main__":
    main()
