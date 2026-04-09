"""
LLM-based structured extractor.

Given raw HTML from a vendor page, ask Claude to return a single JSON
object matching VENDOR_SCHEMA. Using an LLM rather than CSS selectors
makes the extractor robust to site redesigns — we only need URLs to
stay roughly valid.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic

MODEL_ID = "claude-opus-4-6"
MAX_HTML_CHARS = 60_000

VENDOR_SCHEMA = {
    "type": "object",
    "required": ["latest_model", "official_scores"],
    "properties": {
        "latest_model": {
            "type": "object",
            "required": ["display_name"],
            "properties": {
                "id": {"type": "string", "description": "API model id if available"},
                "display_name": {"type": "string", "description": "Marketing name, e.g. 'Claude Opus 4.6'"},
                "release_date": {"type": "string", "description": "YYYY-MM-DD if known, else empty string"},
                "source_url": {"type": "string"},
            },
        },
        "official_scores": {
            "type": "array",
            "description": "Benchmark scores the vendor officially reports for this model",
            "items": {
                "type": "object",
                "required": ["benchmark", "score"],
                "properties": {
                    "benchmark": {"type": "string", "description": "e.g. 'MMLU-Pro', 'GPQA Diamond', 'SWE-bench Verified'"},
                    "score": {"type": "number"},
                    "unit": {"type": "string", "description": "'%' or 'Elo' or '' for raw"},
                },
            },
        },
    },
}

LEADERBOARD_SCHEMA = {
    "type": "object",
    "description": "Mapping from normalised model display name to its score on this leaderboard",
    "additionalProperties": {"type": "number"},
}


_HTML_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _clean_html(html: str) -> str:
    """Drop script/style blocks and collapse whitespace so the LLM sees signal."""
    cleaned = _HTML_TAG_RE.sub(" ", html)
    cleaned = _WS_RE.sub(" ", cleaned)
    return cleaned.strip()[:MAX_HTML_CHARS]


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=api_key)


def _call_tool(system: str, user: str, schema: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Force a structured response via a single-tool tool_use."""
    client = _client()
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=4096,
        system=system,
        tools=[
            {
                "name": tool_name,
                "description": "Return the extracted structured data.",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input  # type: ignore[return-value]
    raise RuntimeError(f"Model did not call tool {tool_name!r}; got: {resp.content!r}")


def extract_vendor(vendor_id: str, flagship_hint: str, html_blobs: list[tuple[str, str]]) -> dict[str, Any]:
    """
    html_blobs: list of (source_url, raw_html) pairs, already fetched.
    Returns a dict conforming to VENDOR_SCHEMA.
    """
    if not html_blobs:
        raise RuntimeError("No HTML fetched for vendor")

    joined = "\n\n".join(
        f"=== SOURCE: {url} ===\n{_clean_html(html)}"
        for url, html in html_blobs
    )

    system = (
        "You extract structured data about the latest flagship AI model released by a specific vendor. "
        "You must only report a model that is officially released (not 'coming soon' or teased). "
        "If multiple models are listed, pick the newest and most capable one matching the flagship hint. "
        "Only include benchmark scores that are explicitly stated on the pages provided. "
        "Do not invent or estimate numbers. If no score is clearly stated, return an empty list."
    )

    user = (
        f"Vendor: {vendor_id}\n"
        f"Flagship hint: {flagship_hint}\n\n"
        f"Page content follows.\n\n{joined}\n\n"
        "Call the `report_latest_model` tool with the result."
    )

    return _call_tool(system, user, VENDOR_SCHEMA, "report_latest_model")


def extract_leaderboard(leaderboard_name: str, model_names: list[str], html: str) -> dict[str, float]:
    """
    Try to pull scores for the given model names from a leaderboard page's HTML.
    Returns a dict: {display_name_from_our_list: score}.
    """
    cleaned = _clean_html(html)
    system = (
        "You extract scores from a public AI benchmark leaderboard page. "
        "For each model name in the input list, find the best matching row on the leaderboard "
        "(fuzzy match on display name / version suffix is OK). "
        "Return a mapping from the input model name (verbatim) to the numeric score shown. "
        "If a model is not on the leaderboard, omit it. Never fabricate scores."
    )
    user = (
        f"Leaderboard: {leaderboard_name}\n"
        f"Models to look up: {json.dumps(model_names, ensure_ascii=False)}\n\n"
        f"Leaderboard page content:\n{cleaned}\n\n"
        "Call the `report_scores` tool."
    )
    return _call_tool(system, user, LEADERBOARD_SCHEMA, "report_scores")  # type: ignore[return-value]
