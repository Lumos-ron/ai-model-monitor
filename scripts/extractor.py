"""
LLM-based structured extractor.

Talks to any OpenAI-compatible chat completions endpoint (currently
defaulting to Kimi / Moonshot AI) and uses OpenAI-style function
calling (`tool_choice`) to force structured JSON output. Two-phase
flow for vendor pages:

    1. discover_latest   — given list/docs pages, identify the current
                           flagship model name and up to 3 candidate
                           URLs for the announcement article.
    2. extract_benchmarks — given the fetched announcement article,
                           pull the official benchmark numbers.

Leaderboard pages still use a single-shot extraction.

Provider is controlled entirely by env vars (LLM_API_KEY /
LLM_BASE_URL / LLM_MODEL) so switching to DeepSeek, Qwen, etc. is
a config change, not a code change.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

MODEL_ID = os.environ.get("LLM_MODEL", "kimi-k2-turbo-preview")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.moonshot.ai/v1")
MAX_HTML_CHARS = int(os.environ.get("LLM_MAX_HTML_CHARS", "30000"))

# ---------------- JSON schemas ----------------

DISCOVERY_SCHEMA = {
    "type": "object",
    "required": ["display_name"],
    "properties": {
        "display_name": {
            "type": "string",
            "description": "Marketing name of the current flagship model, including version suffix (e.g. 'GPT-5', 'Claude Opus 4.6', 'Gemini 3.1 Pro'). Never just a family name like 'MiMo'.",
        },
        "id": {"type": "string", "description": "API model id if shown"},
        "release_date": {"type": "string", "description": "YYYY-MM-DD if known, else ''"},
        "announcement_urls": {
            "type": "array",
            "description": "Up to 3 URLs found on the provided pages that most likely link to the announcement/release/blog post for this specific model. Full URLs only.",
            "items": {"type": "string"},
            "maxItems": 3,
        },
    },
}

BENCHMARK_SCHEMA = {
    "type": "object",
    "required": ["official_scores"],
    "properties": {
        "release_date": {"type": "string"},
        "official_scores": {
            "type": "array",
            "description": "Benchmark scores explicitly stated on the article for this model. Include at most 8 of the most recognised benchmarks (MMLU-Pro, GPQA Diamond, SWE-bench Verified, HumanEval, MATH, AIME, Arena Hard, etc.).",
            "items": {
                "type": "object",
                "required": ["benchmark", "score"],
                "properties": {
                    "benchmark": {"type": "string"},
                    "score": {"type": "number"},
                    "unit": {"type": "string", "description": "'%' for percentages, else ''"},
                },
            },
            "maxItems": 8,
        },
    },
}

LEADERBOARD_SCHEMA = {
    "type": "object",
    "description": "Mapping from input model name to its numeric score. Omit entries for models not on the leaderboard. Never emit null values.",
    "additionalProperties": {"type": "number"},
}

AA_SCORES_SCHEMA = {
    "type": "object",
    "description": (
        "Mapping from input model name (verbatim) to its benchmark scores as shown on the "
        "Artificial Analysis models page. Omit models not found on the page. Omit individual "
        "fields that are not shown for a given model — never emit null values."
    ),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "intelligence_index": {
                "type": "number",
                "description": "AA Intelligence Index — composite score, typically 0-100.",
            },
            "mmlu_pro": {"type": "number", "description": "MMLU-Pro score as shown, percentage 0-100."},
            "gpqa_diamond": {"type": "number", "description": "GPQA Diamond score as shown, percentage 0-100."},
            "humanitys_last_exam": {"type": "number", "description": "Humanity's Last Exam score as shown, percentage 0-100."},
            "livecodebench": {"type": "number", "description": "LiveCodeBench score as shown, percentage 0-100."},
            "scicode": {"type": "number", "description": "SciCode score as shown, percentage 0-100."},
            "math_500": {"type": "number", "description": "MATH-500 score as shown, percentage 0-100."},
            "aime": {"type": "number", "description": "AIME score as shown, percentage 0-100."},
            "ifbench": {"type": "number", "description": "IFBench score as shown, percentage 0-100."},
        },
    },
}


# ---------------- helpers ----------------

_HTML_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _clean(text: str, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else MAX_HTML_CHARS
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = _WS_RE.sub(" ", cleaned)
    return cleaned.strip()[:limit]


def _client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _parse_arguments(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty tool-call arguments")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw)
        if not m:
            raise
        return json.loads(m.group(0))


def _call_tool(system: str, user: str, schema: dict[str, Any], tool_name: str) -> dict[str, Any]:
    client = _client()
    # NOTE: no `temperature` override. Some providers (e.g. Kimi K2.5) reject
    # any value other than 1; we rely on function-calling + strict schema
    # for determinism instead of sampling parameters.
    resp = client.chat.completions.create(
        model=MODEL_ID,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Return the extracted structured data.",
                    "parameters": schema,
                },
            }
        ],
        # NOTE: Kimi K2.5 thinking-mode rejects BOTH the specified form and
        # the "required" literal ("tool_choice 'required' is incompatible
        # with thinking enabled"). We default MODEL_ID to a non-thinking
        # variant (kimi-k2-turbo-preview) where "required" works; other
        # OpenAI-compatible providers also accept it.
        tool_choice="required",
    )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        if tc.function and tc.function.name == tool_name:
            return _parse_arguments(tc.function.arguments)

    content = (msg.content or "").strip()
    if content:
        try:
            return _parse_arguments(content)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Model did not call tool {tool_name!r} and content was not JSON: {exc}"
            ) from exc

    raise RuntimeError(f"Model did not call tool {tool_name!r} and returned no content")


# ---------------- phase 1: discover ----------------

def discover_latest(
    vendor_id: str,
    flagship_hint: str,
    html_blobs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Identify the current flagship model and candidate announcement URLs."""
    if not html_blobs:
        raise RuntimeError("No HTML fetched for vendor")

    joined = "\n\n".join(
        f"=== SOURCE: {url} ===\n{_clean(html)}"
        for url, html in html_blobs
    )

    system = (
        "You read an AI vendor's blog list / docs / GitHub page and identify the current flagship model. "
        "Rules: "
        "(1) Only pick a model that is officially released, not teased. "
        "(2) If multiple versions are listed, pick the newest and most capable one matching the flagship hint. "
        "(3) For announcement_urls, return up to 3 full URLs that appear on the provided pages and most likely "
        "point to the release blog post / model card / technical report for THIS specific model. Do not make up URLs. "
        "(4) You MUST call the `report_flagship` tool."
    )
    user = (
        f"Vendor: {vendor_id}\n"
        f"Flagship hint: {flagship_hint}\n\n"
        f"Pages:\n\n{joined}\n\n"
        "Call the `report_flagship` tool."
    )
    return _call_tool(system, user, DISCOVERY_SCHEMA, "report_flagship")


# ---------------- phase 2: benchmarks ----------------

def extract_benchmarks(
    vendor_id: str,
    display_name: str,
    article_html: str,
    source_url: str,
) -> dict[str, Any]:
    """Pull official benchmark scores from a specific announcement article."""
    cleaned = _clean(article_html)
    system = (
        "You extract benchmark scores from an AI model announcement/article. "
        "Rules: "
        "(1) Only include benchmarks that are explicitly stated in the article for the specified model. "
        "(2) Do not invent, estimate, or carry over numbers from other models mentioned for comparison. "
        "(3) Prefer the pass@1 / main reported number when multiple variants are shown. "
        "(4) If no scores are clearly stated for this model, return an empty list. "
        "(5) You MUST call the `report_benchmarks` tool."
    )
    user = (
        f"Vendor: {vendor_id}\n"
        f"Model: {display_name}\n"
        f"Source URL: {source_url}\n\n"
        f"Article content:\n{cleaned}\n\n"
        "Call the `report_benchmarks` tool."
    )
    return _call_tool(system, user, BENCHMARK_SCHEMA, "report_benchmarks")


# ---------------- leaderboards ----------------

def extract_aa_scores(
    model_names: list[str],
    html: str,
) -> dict[str, dict[str, Any]]:
    """Pull per-model benchmark scores from the Artificial Analysis models page.

    AA aggregates normalised benchmark results across every major vendor
    (OpenAI / Anthropic / Google / DeepSeek / Zhipu / Xiaomi / MiniMax), so a
    single fetch + single LLM call covers all 7 vendors. The AA page is larger
    than a blog post, so we give it a bigger character budget than the
    global default.
    """
    # AA's models page has a big table — give it more room than the default.
    cleaned = _clean(html, max_chars=max(MAX_HTML_CHARS, 80000))
    system = (
        "You extract benchmark scores from the Artificial Analysis models page "
        "(artificialanalysis.ai/models), which aggregates standardised benchmark "
        "results for every major AI model. "
        "For each model name in the input list, find the best matching row on the "
        "page (fuzzy match on version suffix is OK) and return the numeric scores "
        "shown there. "
        "Rules: "
        "(1) Return the number exactly as shown. If the page shows '87%' return 87, not 0.87. "
        "(2) If a model is not on the page, omit its key entirely — never return null. "
        "(3) If a specific metric is missing for a model, omit that field — never return null. "
        "(4) Never fabricate scores; if uncertain, omit. "
        "(5) Numbers must be numbers, not strings. "
        "(6) You MUST call the `report_aa_scores` tool."
    )
    user = (
        f"Models to look up: {json.dumps(model_names, ensure_ascii=False)}\n\n"
        f"Artificial Analysis page content:\n{cleaned}\n\n"
        "Call the `report_aa_scores` tool."
    )
    return _call_tool(system, user, AA_SCORES_SCHEMA, "report_aa_scores")  # type: ignore[return-value]


def extract_leaderboard(leaderboard_name: str, model_names: list[str], html: str) -> dict[str, float]:
    cleaned = _clean(html)
    system = (
        "You extract scores from a public AI benchmark leaderboard page. "
        "For each model name in the input list, find the best matching row (fuzzy match on version suffix is OK). "
        "Return a mapping from the input model name (verbatim) to the numeric score shown. "
        "Rules: "
        "(1) If a model is not on the leaderboard, omit its key — never return null. "
        "(2) The score must be a number, not a string. "
        "(3) Never fabricate scores. "
        "(4) You MUST call the `report_scores` tool."
    )
    user = (
        f"Leaderboard: {leaderboard_name}\n"
        f"Models to look up: {json.dumps(model_names, ensure_ascii=False)}\n\n"
        f"Leaderboard page content:\n{cleaned}\n\n"
        "Call the `report_scores` tool."
    )
    return _call_tool(system, user, LEADERBOARD_SCHEMA, "report_scores")  # type: ignore[return-value]
