"""
Scrape authoritative reference material refuting antisemitic tropes.

Unlike `rag.scrape_cmv_israel`, which harvests delta-awarded Reddit comments
(persuasive, but with no guarantee of factual accuracy), this module pulls the
*factual record* from curated authoritative sources. That record is what the
graph's groundedness gate (`hallucination_check`) checks a draft's claims
against, so the two corpora play different roles: CMV supplies rhetoric that
demonstrably moved a human, reference supplies verifiable facts.

Two sources, both access-checked before use:
  - USHMM Holocaust Encyclopedia (encyclopedia.ushmm.org) — robots.txt permits
    everything except /search. Best coverage of Holocaust denial specifically;
    the slug list below is taken from its "Holocaust Denial" series index.
  - Wikipedia action API — a public API, so no scraping etiquette concerns.
    Best coverage of the remaining tropes (blood libel, the Protocols, the
    Khazar myth, Great Replacement, dual loyalty, banking conspiracies).

Every article title/slug here was verified to resolve and return substantive
body text; see git history for the probe.

Output (the downstream rag.ingest_reference step reads the parquet):
    data/trope_reference.parquet  /  .jsonl

Usage:
    uv run python -m rag.scrape_reference
    uv run python -m rag.scrape_reference --smoke   # 2 docs per source, no write
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date, timezone, datetime
from pathlib import Path

import pandas as pd
import requests

from schemas import ReferenceDocument

USER_AGENT = "argument-quality-research/0.1 (academic use; see repo)"
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 1.5          # be a polite citizen on both hosts
MAX_RETRIES = 4
RETRY_BACKOFF = 3.0
MIN_BODY_CHARS = 500         # below this, treat as a stub/error page and skip

OUT_DIR = Path("data")
OUT_PARQUET = OUT_DIR / "trope_reference.parquet"
OUT_JSONL = OUT_DIR / "trope_reference.jsonl"

WIKI_API = "https://en.wikipedia.org/w/api.php"
USHMM_ARTICLE = "https://encyclopedia.ushmm.org/content/en/article/{slug}"
ADL_ARTICLE = "https://www.adl.org/resources/backgrounders/{slug}"

# --- Source catalogue ------------------------------------------------------
# Maps each documented trope to the articles that refute it. The trope key is
# carried into metadata so retrieval can filter by trope once the harvester's
# classifier has named one.

WIKIPEDIA_ARTICLES: dict[str, list[str]] = {
    "blood_libel": ["Blood libel", "Simon of Trent", "Damascus affair"],
    "protocols": ["The Protocols of the Elders of Zion", "Maurice Joly"],
    "holocaust_denial": [
        "Holocaust denial",
        "Evidence and documentation for the Holocaust",
        "Irving v Penguin Books Ltd",
    ],
    "khazar_myth": [
        "Khazar hypothesis of Ashkenazi ancestry",
        "Genetic studies of Jews",
    ],
    # NB: "Rothschild family" is family history, not a debunking. On its own it
    # left hallucination_check with nothing to verify a refutation against (the
    # grader flagged the banking claims as unsupported), so the conspiracy
    # articles below carry the actual refutation.
    "banking_conspiracy": [
        "Rothschild family",
        "Economic antisemitism",
        "New World Order conspiracy theory",
        "Judeo-Masonic conspiracy theory",
        "Kosher tax conspiracy theory",
    ],
    "great_replacement": ["Great Replacement conspiracy theory", "Eurabia conspiracy theory"],
    "dual_loyalty": ["Dual loyalty"],
    "media_control": [
        "Jewish Bolshevism",
        "Antisemitic trope",
        "Zionist Occupation Government conspiracy theory",
    ],
    "religious_tropes": ["Jewish deicide", "Jewish question"],
}

# Slugs taken from the encyclopedia's "Holocaust Denial" series index.
# ADL backgrounders. robots.txt permits /resources/backgrounders/ (only
# /resources/search/research-analysis is disallowed). Kept deliberately small:
# the slug below was verified to return 200 with substantive prose.
ADL_ARTICLES: dict[str, list[str]] = {
    "banking_conspiracy": [
        "jewish-control-of-the-federal-reserve-a-classic-antisemitic-myth",
    ],
}

USHMM_ARTICLES: dict[str, list[str]] = {
    "holocaust_denial": [
        "holocaust-denial-key-dates",
        "combating-holocaust-denial-origins-of-holocaust-denial",
        "combating-holocaust-denial-evidence-of-the-holocaust-presented-at-nuremberg",
        "holocaust-deniers-and-public-misinformation",
        "evidence-from-the-holocaust",
        "john-demjanjuk-prosecution-of-a-nazi-collaborator",
    ],
    "general_antisemitism": ["antisemitism"],
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get(session: requests.Session, url: str, params: dict | None = None):
    """GET with retries. Returns the Response, or None if it never succeeded."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429, 503):
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return None


# --- Wikipedia -------------------------------------------------------------

def fetch_wikipedia(session: requests.Session, title: str) -> tuple[str, str] | None:
    """Return (resolved_title, plaintext) for one article, or None.

    One title per request: the action API returns a full extract only for the
    first page when several titles are batched.
    """
    r = _get(session, WIKI_API, {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "redirects": 1, "titles": title,
    })
    if r is None:
        return None
    try:
        pages = r.json().get("query", {}).get("pages", {})
    except ValueError:
        return None
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        return None
    text = (page.get("extract") or "").strip()
    return (page.get("title") or title, text) if text else None


# --- USHMM -----------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
# The encyclopedia wraps article prose in a #main / <article> region; grabbing
# it keeps nav, footer, and related-content boilerplate out of the embedding.
_ARTICLE_RE = re.compile(r"<article\b.*?</article>", re.S | re.I)


def _html_to_text(html: str) -> str:
    """Strip tags and collapse whitespace, preserving paragraph breaks."""
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    html = re.sub(r"</p\s*>|<br\s*/?>|</h[1-6]\s*>", "\n\n", html, flags=re.I)
    text = _TAG_RE.sub(" ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#39;", "'").replace("&quot;", '"')
                .replace("&ldquo;", '"').replace("&rdquo;", '"')
                .replace("&lsquo;", "'").replace("&rsquo;", "'")
                .replace("&mdash;", "—").replace("&ndash;", "–"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


# Chrome the <article> region still contains: the image lightbox, the share /
# cite / print controls, the tag list, and the "available in these languages"
# menu. All of it sits BEFORE the prose, so we drop everything up to the last
# marker that appears, then clean up any stragglers line-by-line.
_USHMM_PREAMBLE_MARKERS = (
    "Close language menu",
    "View this term in the glossary",
    "This content is available in the following languages",
    "Skip complementary section (continue reading)",
)
_USHMM_CHROME_LINES = re.compile(
    r"^\s*(More information about this image|Close Image Lightbox|Cite|Share|Print"
    r"|Tags|Language|English|Skip complementary section.*|View this term in the glossary"
    r"|Close language menu|This content is available.*)\s*$",
    re.I,
)


def _strip_ushmm_chrome(text: str) -> str:
    """Drop the nav/share/language boilerplate that precedes the article prose."""
    cut = 0
    for marker in _USHMM_PREAMBLE_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            cut = max(cut, idx + len(marker))
    body = text[cut:] if cut else text
    lines = [ln for ln in body.split("\n") if not _USHMM_CHROME_LINES.match(ln)]
    return re.sub(r"\n\s*\n\s*", "\n\n", "\n".join(lines)).strip()


def fetch_html_article(session: requests.Session, url: str,
                       strip_chrome: bool = True) -> tuple[str, str] | None:
    """Return (title, plaintext) for one HTML article page, or None.

    Shared by the USHMM and ADL arms: both render prose inside <article>, and
    both wrap it in nav/share chrome that `_strip_ushmm_chrome` removes.
    """
    r = _get(session, url)
    if r is None:
        return None
    html = r.text
    m = _TITLE_RE.search(html)
    title = _html_to_text(m.group(1)) if m else url.rsplit("/", 1)[-1]
    title = title.split("|")[0].strip() or url.rsplit("/", 1)[-1]
    body_html = "\n".join(m.group(0) for m in _ARTICLE_RE.finditer(html)) or html
    text = _html_to_text(body_html)
    if strip_chrome:
        text = _strip_ushmm_chrome(text)
    return (title, text) if len(text) >= MIN_BODY_CHARS else None


def fetch_ushmm(session: requests.Session, slug: str) -> tuple[str, str] | None:
    """Return (title, plaintext) for one encyclopedia article, or None."""
    r = _get(session, USHMM_ARTICLE.format(slug=slug))
    if r is None:
        return None
    html = r.text
    m = _TITLE_RE.search(html)
    title = _html_to_text(m.group(1)) if m else slug
    title = title.split("|")[0].strip() or slug
    body_html = "\n".join(m.group(0) for m in _ARTICLE_RE.finditer(html)) or html
    text = _strip_ushmm_chrome(_html_to_text(body_html))
    return (title, text) if len(text) >= MIN_BODY_CHARS else None


# --- Orchestration ---------------------------------------------------------

def collect(limit_per_source: int | None = None) -> list[ReferenceDocument]:
    """Fetch every catalogued article and return ReferenceDocument records."""
    session = _session()
    today = datetime.now(timezone.utc).date()
    docs: list[ReferenceDocument] = []

    print("Fetching Wikipedia articles...")
    n = 0
    for trope, titles in WIKIPEDIA_ARTICLES.items():
        for title in titles:
            if limit_per_source is not None and n >= limit_per_source:
                break
            got = fetch_wikipedia(session, title)
            time.sleep(SLEEP_BETWEEN)
            if not got:
                print(f"  [skip] wikipedia:{title}")
                continue
            resolved, text = got
            docs.append(ReferenceDocument(
                doc_id=f"wikipedia::{resolved.replace(' ', '_')}",
                source="wikipedia",
                trope=trope,
                title=resolved,
                url=f"https://en.wikipedia.org/wiki/{resolved.replace(' ', '_')}",
                text=text,
                retrieved=today,
            ))
            n += 1
            print(f"  [ok]   {len(text):>7} chars  {resolved}")

    print("\nFetching USHMM Holocaust Encyclopedia articles...")
    n = 0
    for trope, slugs in USHMM_ARTICLES.items():
        for slug in slugs:
            if limit_per_source is not None and n >= limit_per_source:
                break
            got = fetch_ushmm(session, slug)
            time.sleep(SLEEP_BETWEEN)
            if not got:
                print(f"  [skip] ushmm:{slug}")
                continue
            title, text = got
            docs.append(ReferenceDocument(
                doc_id=f"ushmm::{slug}",
                source="ushmm",
                trope=trope,
                title=title,
                url=USHMM_ARTICLE.format(slug=slug),
                text=text,
                retrieved=today,
            ))
            n += 1
            print(f"  [ok]   {len(text):>7} chars  {title}")

    print("\nFetching ADL backgrounders...")
    n = 0
    for trope, slugs in ADL_ARTICLES.items():
        for slug in slugs:
            if limit_per_source is not None and n >= limit_per_source:
                break
            got = fetch_html_article(session, ADL_ARTICLE.format(slug=slug))
            time.sleep(SLEEP_BETWEEN)
            if not got:
                print(f"  [skip] adl:{slug}")
                continue
            title, text = got
            docs.append(ReferenceDocument(
                doc_id=f"adl::{slug}",
                source="adl",
                trope=trope,
                title=title,
                url=ADL_ARTICLE.format(slug=slug),
                text=text,
                retrieved=today,
            ))
            n += 1
            print(f"  [ok]   {len(text):>7} chars  {title}")

    return docs


def main(smoke: bool = False) -> None:
    """Fetch the catalogue and write parquet + jsonl (skipped for --smoke)."""
    docs = collect(limit_per_source=2 if smoke else None)
    print(f"\nCollected {len(docs)} reference documents.")
    if smoke:
        print("[smoke] no files written.")
        return
    if not docs:
        print("Nothing collected; not writing.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame([d.model_dump() for d in docs])
    df["retrieved"] = pd.to_datetime(df["retrieved"])
    df.to_parquet(OUT_PARQUET, index=False)
    OUT_JSONL.write_text("".join(d.model_dump_json() + "\n" for d in docs))
    print(f"Saved {len(df)} documents to {OUT_PARQUET} and {OUT_JSONL}.")
    print("\nPer-trope counts:")
    print(df["trope"].value_counts().to_string())


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Fetch 2 docs per source and write nothing")
    args = parser.parse_args()
    main(smoke=args.smoke)


if __name__ == "__main__":
    cli()
