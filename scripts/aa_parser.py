"""
Direct (no-LLM) parser for Artificial Analysis model data.

AA's /models page is a Next.js App Router page that streams its data
through React Server Components as `self.__next_f.push([1, "..."])`
Flight payload chunks. Rather than fire up a headless browser or pay
an LLM round-trip to parse the rendered tables, we extract the raw
Flight chunks with a regex, `unicode_escape`-decode the JS string,
and scan the result for the benchmark fields we care about.

Per-model object layout in the Flight payload (simplified):
    {..., "mmlu_pro": 0.845, "gpqa": 0.712, ..., "intelligence_index": 86.2,
     "name": "GPT-5 (high)", "is_open_weights": false, ...}

Score scale:
- intelligence_index is already 0–100
- every other benchmark (mmlu_pro, gpqa, hle, livecodebench, scicode,
  math_500, aime, ifbench) is stored as a 0.0–1.0 ratio; we scale to %.
"""

from __future__ import annotations

import re

import requests

AA_MODELS_URL = "https://artificialanalysis.ai/models"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
FETCH_TIMEOUT = 60

# AA's internal field names → what we'll store in models.json
AA_FIELD_TO_LABEL: dict[str, str] = {
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA Diamond",
    "hle": "Humanity's Last Exam",
    "livecodebench": "LiveCodeBench",
    "scicode": "SciCode",
    "math_500": "MATH-500",
    "aime": "AIME 2025",
    "ifbench": "IFBench",
}

# Benchmark fields stored as 0.0–1.0 ratios on AA (to be ×100 for %).
AA_RATIO_FIELDS = set(AA_FIELD_TO_LABEL.keys())

# Scalar field (0–100) lifted into third_party_scores.
AA_INDEX_FIELD = "intelligence_index"

_ALL_FIELDS = list(AA_FIELD_TO_LABEL.keys()) + [AA_INDEX_FIELD]

_NEXT_F_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', re.DOTALL)
_ANCHOR_RE = re.compile(r'"name":"([^"]+)","is_open_weights"')
_SCORE_RE = re.compile(r'"(' + "|".join(_ALL_FIELDS) + r')":(null|-?[0-9.]+)')


def fetch_aa_models_html() -> str:
    resp = requests.get(
        AA_MODELS_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=FETCH_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text


def parse_flight_payload(html: str) -> str:
    """Concatenate every __next_f.push chunk and decode JS string escapes."""
    parts = _NEXT_F_RE.findall(html)
    if not parts:
        return ""
    joined = "".join(parts)
    # The chunks are JS-literal-escaped strings — `\"`, `\u002F`, etc.
    # unicode_escape handles all of those in one pass.
    return joined.encode("utf-8", errors="replace").decode("unicode_escape", errors="replace")


def extract_model_scores(flight_payload: str) -> dict[str, dict[str, float]]:
    """Walk the payload anchor-by-anchor.

    Each model object ends with `"name":"...","is_open_weights":...`, so
    we slice the payload between consecutive anchor positions and hunt
    for known score fields inside each slice. This tolerates unknown
    keys, nested objects, and field-ordering changes as long as the
    `name`/`is_open_weights` pair stays the terminating anchor."""
    out: dict[str, dict[str, float]] = {}
    prev_end = 0
    for m in _ANCHOR_RE.finditer(flight_payload):
        segment = flight_payload[prev_end:m.start()]
        prev_end = m.end()
        scores: dict[str, float] = {}
        for sm in _SCORE_RE.finditer(segment):
            field = sm.group(1)
            raw = sm.group(2)
            if raw == "null":
                continue
            try:
                num = float(raw)
            except ValueError:
                continue
            if field in AA_RATIO_FIELDS:
                num *= 100.0
            scores[field] = round(num, 2)
        if scores:
            out[m.group(1)] = scores
    return out


_PUNCT_RE = re.compile(r"[^a-z0-9]+")

# Size/tier tokens we must NOT allow to appear right after the target name,
# because they indicate a cheaper sibling model (e.g. "Claude Opus" must not
# match "Claude Opus Haiku"). Variant words that indicate evaluation modes
# like "reasoning", "xhigh", "non-reasoning" are intentionally *allowed*.
_SIZE_TIER_TOKENS = {
    "nano", "mini", "micro", "small", "air", "lite", "flash",
    "haiku", "sonnet", "pro", "ultra", "max", "omni", "codex",
    "v", "exp",  # "V3.2 Exp", "GLM-4.5V"
}


def _tokens(s: str) -> list[str]:
    """Normalize to lowercase alphanumeric tokens.

    "GPT-5.4" → ["gpt", "5", "4"]; "MiniMax-M2.7" → ["minimax", "m2", "7"].
    """
    return [t for t in _PUNCT_RE.split(s.lower()) if t]


def best_variant(display_name: str, aa_models: dict[str, dict[str, float]]) -> str | None:
    """Fuzzy-match one of our vendor flagships to an AA row.

    Rules:
    1. Tokenize both names on non-alphanumerics. The AA row must START
       with the display_name's tokens (prefix match).
    2. The token immediately after the prefix must not be a numeric
       version extension (would indicate a different version — e.g.
       target "GPT-5" should not eat "GPT-5.4") nor a size tier word
       (e.g. "nano", "mini", "air") unless that word was already part
       of the target.
    3. Among survivors pick the highest intelligence_index; tie-break
       on shorter name so "GLM-4.5 (Reasoning)" beats any longer alias.
    """
    if not display_name:
        return None
    target = _tokens(display_name)
    if not target:
        return None
    tlen = len(target)

    candidates: list[tuple[str, dict[str, float]]] = []
    for name, scores in aa_models.items():
        cand = _tokens(name)
        if cand[:tlen] != target:
            continue
        if len(cand) > tlen:
            next_tok = cand[tlen]
            if next_tok.isdigit():
                continue
            if next_tok in _SIZE_TIER_TOKENS:
                continue
        candidates.append((name, scores))

    if not candidates:
        return None

    def rank(kv: tuple[str, dict[str, float]]) -> tuple[float, int]:
        idx = kv[1].get(AA_INDEX_FIELD)
        return (-(idx if isinstance(idx, (int, float)) else 0.0), len(kv[0]))

    candidates.sort(key=rank)
    return candidates[0][0]


def get_aa_scores() -> dict[str, dict[str, float]]:
    """High-level entry point: fetch → parse → return {model_name: scores}."""
    html = fetch_aa_models_html()
    payload = parse_flight_payload(html)
    return extract_model_scores(payload)
