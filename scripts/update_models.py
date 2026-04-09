#!/usr/bin/env python3
"""
Orchestrator that refreshes data/models.json.

Flow:
    1. Load previous data (if any) as a fallback.
    2. For each vendor, fetch configured URLs and ask Claude to extract
       {latest_model, official_scores}. On failure, keep the previous
       vendor block and mark fetch_status=error.
    3. Fetch each third-party leaderboard once, ask Claude to pull the
       score for each vendor's latest model.
    4. Merge and write data/models.json.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import requests

from extractor import extract_leaderboard, extract_vendor
from sources import LEADERBOARDS, VENDORS, Leaderboard, Vendor

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "models.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
FETCH_TIMEOUT = 30


def fetch(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT)
    resp.raise_for_status()
    return resp.text


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
            "source_url": "",
        },
        "official_scores": [],
        "third_party_scores": {},
        "fetch_status": "error",
        "fetch_error": "No data yet",
    }


def update_vendor(vendor: Vendor, prev: dict[str, Any]) -> dict[str, Any]:
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
        return block

    try:
        extracted = extract_vendor(vendor.id, vendor.flagship_hint, html_blobs)
    except Exception as exc:  # noqa: BLE001
        print(f"[{vendor.id}] extractor failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        block = copy.deepcopy(prev_block)
        block["fetch_status"] = "error"
        block["fetch_error"] = f"extractor: {exc}"
        return block

    latest = extracted.get("latest_model") or {}
    # If extractor did not pick a source_url, fall back to the first fetched URL.
    if not latest.get("source_url"):
        latest["source_url"] = html_blobs[0][0]

    block = {
        "id": vendor.id,
        "name_zh": vendor.name_zh,
        "name_en": vendor.name_en,
        "product": vendor.product,
        "latest_model": {
            "id": latest.get("id", ""),
            "display_name": latest.get("display_name", ""),
            "release_date": latest.get("release_date", ""),
            "source_url": latest.get("source_url", ""),
        },
        "official_scores": extracted.get("official_scores", []) or [],
        # third_party_scores populated in a later pass; carry over previous
        # values for now so we never lose them if a leaderboard fetch fails.
        "third_party_scores": dict(prev_block.get("third_party_scores", {})),
        "fetch_status": "ok",
        "fetch_error": None,
    }
    return block


def update_leaderboard_scores(vendor_blocks: list[dict[str, Any]]) -> None:
    """Mutates vendor_blocks in place, filling third_party_scores."""
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

        for vb in vendor_blocks:
            name = vb["latest_model"].get("display_name")
            if not name:
                continue
            if name in scores:
                vb["third_party_scores"][lb.score_field] = scores[name]


def main() -> int:
    prev = load_previous()
    vendor_blocks: list[dict[str, Any]] = []

    for vendor in VENDORS:
        print(f"[{vendor.id}] updating…", file=sys.stderr)
        vendor_blocks.append(update_vendor(vendor, prev))

    print("Updating third-party leaderboards…", file=sys.stderr)
    update_leaderboard_scores(vendor_blocks)

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
