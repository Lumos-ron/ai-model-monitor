#!/usr/bin/env python3
"""
Orchestrator that refreshes data/models.json.

Flow per vendor:
    1. Fetch the configured list/docs pages (with r.jina.ai fallback if
       a source returns 401/403/429 — OpenAI is WAF'd from GH runners).
    2. `discover_latest`: LLM picks the flagship model + up to 3
       candidate announcement URLs.
    3. For each candidate URL, fetch it and call `extract_benchmarks`.
       Keep the first result that returns a non-empty official_scores
       list. Fall back to empty scores if none succeeds.
    4. Merge with previous vendor block; on any hard failure we keep
       the previous data and mark fetch_status=error.

Then a single pass over the third-party leaderboards fills
third_party_scores for every vendor, with null filtering and
string→number coercion.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from aa_parser import (
    AA_FIELD_TO_LABEL,
    AA_INDEX_FIELD,
    best_variant,
    get_aa_scores,
)
from extractor import discover_latest, extract_benchmarks, extract_leaderboard
from sources import LEADERBOARDS, VENDORS, Vendor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "models.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
FETCH_TIMEOUT = 30
JINA_READER_PREFIX = "https://r.jina.ai/"
WAF_STATUSES = {401, 403, 429, 503}


# ---------------- fetching ----------------

def _fetch_direct(url: str) -> str:
    resp = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text


def _fetch_via_reader(url: str) -> str:
    """Route through r.jina.ai — a free HTML-to-markdown proxy that
    uses a headless browser upstream, so it bypasses most WAFs and
    returns clean text (fewer tokens for the extractor)."""
    reader_url = JINA_READER_PREFIX + url
    resp = requests.get(
        reader_url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/plain"},
        timeout=FETCH_TIMEOUT * 2,
    )
    resp.raise_for_status()
    return resp.text


def fetch(url: str) -> str:
    """Direct fetch with automatic Jina reader fallback on WAF blocks."""
    try:
        return _fetch_direct(url)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in WAF_STATUSES:
            print(f"  direct fetch {url} → {status}, retrying via r.jina.ai", file=sys.stderr)
            return _fetch_via_reader(url)
        raise
    except requests.RequestException as exc:
        # Connection / SSL / timeout issues — one retry via reader.
        print(f"  direct fetch {url} → {exc}, retrying via r.jina.ai", file=sys.stderr)
        return _fetch_via_reader(url)


# ---------------- data loading ----------------

def load_previous() -> dict[str, Any]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"last_updated": None, "vendors": []}


def previous_vendor(prev: dict[str, Any], vendor_id: str) -> dict[str, Any] | None:
    for v in prev.get("vendors", []):
        if v.get("id") == vendor_id:
            return v
    return None


def empty_vendor_block(vendor: Vendor) -> dict[str, Any]:
    return {
        "id": vendor.id,
        "name_zh": vendor.name_zh,
        "name_en": vendor.name_en,
        "product": vendor.product,
        "latest_model": {
            "id": "",
            "display_name": "",
            "release_date": "",
            "source_url": vendor.urls[0] if vendor.urls else "",
        },
        "official_scores": [],
        "third_party_scores": {},
        "fetch_status": "error",
        "fetch_error": "No data yet",
    }


def _is_http_url(candidate: str) -> bool:
    try:
        p = urlparse(candidate)
    except ValueError:
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


# ---------------- vendor discovery (phase A) ----------------

def discover_vendor(
    vendor: Vendor, prev: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Phase A: fetch vendor seed pages and identify the flagship model.

    Returns (vendor_block_with_empty_official_scores, candidate_announcement_urls).
    The candidate URLs are stashed for the phase C fallback — they are only
    walked if Artificial Analysis doesn't have the model.
    """
    prev_block = previous_vendor(prev, vendor.id) or empty_vendor_block(vendor)

    html_blobs: list[tuple[str, str]] = []
    fetch_errors: list[str] = []
    for url in vendor.urls:
        try:
            html_blobs.append((url, fetch(url)))
        except Exception as exc:  # noqa: BLE001
            fetch_errors.append(f"{url}: {exc}")

    if not html_blobs:
        block = copy.deepcopy(prev_block)
        block["fetch_status"] = "error"
        block["fetch_error"] = "; ".join(fetch_errors) or "no sources fetched"
        block["third_party_scores"] = _sanitize_third_party(block.get("third_party_scores"))
        return block, []

    try:
        discovery = discover_latest(vendor.id, vendor.flagship_hint, html_blobs)
    except Exception as exc:  # noqa: BLE001
        print(f"[{vendor.id}] discover failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        block = copy.deepcopy(prev_block)
        block["fetch_status"] = "error"
        block["fetch_error"] = f"discover: {exc}"
        block["third_party_scores"] = _sanitize_third_party(block.get("third_party_scores"))
        return block, []

    display_name = (discovery.get("display_name") or "").strip()
    release_date = (discovery.get("release_date") or "").strip()
    model_id = (discovery.get("id") or "").strip()
    candidate_urls = [
        u for u in (discovery.get("announcement_urls") or [])
        if isinstance(u, str) and _is_http_url(u)
    ]

    if not display_name:
        block = copy.deepcopy(prev_block)
        block["fetch_status"] = "error"
        block["fetch_error"] = "discovery returned empty display_name"
        block["third_party_scores"] = _sanitize_third_party(block.get("third_party_scores"))
        return block, []

    source_url = candidate_urls[0] if candidate_urls else html_blobs[0][0]
    block = {
        "id": vendor.id,
        "name_zh": vendor.name_zh,
        "name_en": vendor.name_en,
        "product": vendor.product,
        "latest_model": {
            "id": model_id,
            "display_name": display_name,
            "release_date": release_date,
            "source_url": source_url,
        },
        "official_scores": [],
        "third_party_scores": _sanitize_third_party(prev_block.get("third_party_scores")),
        "fetch_status": "ok",
        "fetch_error": None,
    }
    return block, candidate_urls


# ---------------- AA benchmark pass (phase B) ----------------

def apply_aa_benchmarks(vendor_blocks: list[dict[str, Any]]) -> set[str]:
    """Phase B: pull per-benchmark scores + AA Intelligence Index from
    Artificial Analysis's /models page in a single request.

    AA is a Next.js App Router site whose RSC Flight payload embeds the
    full scores table as inline JSON. `aa_parser` decodes that payload
    with a regex — no headless browser, no LLM — and returns
    {model_name: {field: score, ...}}. We then fuzzy-match each vendor's
    discovered display_name to one of AA's variants (preferring the
    reasoning mode with the highest intelligence_index).

    Scores go into `official_scores` in the canonical order given by
    AA_FIELD_TO_LABEL; AA's `intelligence_index` is lifted directly into
    `third_party_scores` as a bonus.

    Returns the set of vendor ids that got at least one AA benchmark
    score, so the caller can skip the per-vendor announcement-page
    fallback for them.
    """
    filled: set[str] = set()
    try:
        aa_models = get_aa_scores()
    except Exception as exc:  # noqa: BLE001
        print(f"[AA] fetch/parse failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return filled

    if not aa_models:
        print("[AA] no models parsed from Flight payload", file=sys.stderr)
        return filled

    print(f"[AA] parsed {len(aa_models)} models from flight payload", file=sys.stderr)

    for vb in vendor_blocks:
        display_name = vb["latest_model"].get("display_name") or ""
        match_name = best_variant(display_name, aa_models)
        if not match_name:
            print(f"[AA] no variant matched for {display_name!r}", file=sys.stderr)
            continue

        match_scores = aa_models[match_name]
        print(f"[AA] {display_name!r} → {match_name!r}", file=sys.stderr)

        official: list[dict[str, Any]] = []
        for field, label in AA_FIELD_TO_LABEL.items():
            raw = match_scores.get(field)
            num = _coerce_score(raw)
            if num is None:
                continue
            official.append({"benchmark": label, "score": num, "unit": "%"})

        if official:
            vb["official_scores"] = official
            vb["fetch_status"] = "ok"
            vb["fetch_error"] = None
            filled.add(vb["id"])

        idx = _coerce_score(match_scores.get(AA_INDEX_FIELD))
        if idx is not None:
            vb["third_party_scores"]["aa_intelligence_index"] = idx

    return filled


# ---------------- fallback benchmarks (phase C) ----------------

def fallback_benchmarks(block: dict[str, Any], candidate_urls: list[str]) -> None:
    """Phase C: for vendors AA didn't cover, walk the discovery-produced
    candidate announcement URLs and pull whatever benchmarks the article
    explicitly lists. First URL that returns a non-empty score list wins."""
    if not candidate_urls:
        return

    display_name = block["latest_model"]["display_name"]
    vendor_id = block["id"]
    notes: list[str] = []

    for cand in candidate_urls:
        try:
            article_html = fetch(cand)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"fetch {cand}: {exc}")
            continue
        try:
            result = extract_benchmarks(vendor_id, display_name, article_html, cand)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"extract {cand}: {exc}")
            continue

        scores = result.get("official_scores") or []
        if scores:
            block["official_scores"] = scores
            block["latest_model"]["source_url"] = cand
            if not block["latest_model"].get("release_date"):
                block["latest_model"]["release_date"] = (result.get("release_date") or "").strip()
            return
        notes.append(f"{cand}: no scores found")

    if notes and not block.get("fetch_error"):
        block["fetch_error"] = "no benchmarks found: " + "; ".join(notes)


# ---------------- leaderboards ----------------

def _coerce_score(raw: Any) -> float | None:
    """MiMo occasionally returns numbers as strings or sneaks in null.
    Normalise everything to a finite float or drop it."""
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool is a subclass of int — reject explicitly
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().rstrip("%")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _sanitize_third_party(scores: dict[str, Any] | None) -> dict[str, float]:
    """Drop nulls and coerce stringified numbers when we inherit the
    previous vendor block. Runs regardless of whether the current
    leaderboard pass succeeds, so stale junk from earlier runs can't
    survive indefinitely."""
    out: dict[str, float] = {}
    for k, v in (scores or {}).items():
        num = _coerce_score(v)
        if num is not None:
            out[k] = num
    return out


def update_leaderboard_scores(vendor_blocks: list[dict[str, Any]]) -> None:
    model_names = [
        vb["latest_model"]["display_name"]
        for vb in vendor_blocks
        if vb["latest_model"].get("display_name")
    ]
    if not model_names:
        return

    for lb in LEADERBOARDS:
        try:
            html = fetch(lb.url)
            scores = extract_leaderboard(lb.name, model_names, html)
        except Exception as exc:  # noqa: BLE001
            print(f"[leaderboard:{lb.id}] failed: {exc}", file=sys.stderr)
            continue

        if not isinstance(scores, dict):
            print(f"[leaderboard:{lb.id}] unexpected response type {type(scores)}", file=sys.stderr)
            continue

        for vb in vendor_blocks:
            name = vb["latest_model"].get("display_name")
            if not name or name not in scores:
                continue
            num = _coerce_score(scores[name])
            if num is None:
                # Drop any stale value for this field rather than leave
                # a nonsense cached number.
                vb["third_party_scores"].pop(lb.score_field, None)
                continue
            vb["third_party_scores"][lb.score_field] = num


# ---------------- main ----------------

def main() -> int:
    prev = load_previous()
    vendor_blocks: list[dict[str, Any]] = []
    vendor_candidates: dict[str, list[str]] = {}

    # Phase A: discover each vendor's flagship model
    for vendor in VENDORS:
        print(f"[{vendor.id}] discovering…", file=sys.stderr)
        block, candidates = discover_vendor(vendor, prev)
        vendor_blocks.append(block)
        vendor_candidates[vendor.id] = candidates

    # Phase B: pull per-benchmark scores for every vendor from the AA
    # evaluation leaderboards (one page per benchmark).
    filled = apply_aa_benchmarks(vendor_blocks)

    # Phase C: for any vendor AA didn't cover, fall back to the vendor's
    # own announcement pages.
    for vb in vendor_blocks:
        if vb["id"] in filled or vb.get("fetch_status") == "error":
            continue
        if vb["official_scores"]:
            continue
        print(f"[{vb['id']}] AA miss — falling back to announcement pages…", file=sys.stderr)
        fallback_benchmarks(vb, vendor_candidates.get(vb["id"], []))

    # Phase D: other third-party leaderboards (LMArena, LiveBench)
    print("Updating third-party leaderboards…", file=sys.stderr)
    update_leaderboard_scores(vendor_blocks)

    # Final sanitize — covers error-path blocks that inherited stale
    # third_party_scores via deepcopy without going through a score pass.
    for vb in vendor_blocks:
        vb["third_party_scores"] = _sanitize_third_party(vb.get("third_party_scores"))

    payload = {
        "last_updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vendors": vendor_blocks,
    }
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {DATA_FILE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
