"""
Configuration for AI vendor sources and third-party leaderboards.

Each vendor entry lists one or more URLs whose HTML should be fetched
and handed to the LLM extractor. The extractor is told which vendor
it is looking at and returns structured JSON.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Vendor:
    id: str
    name_zh: str
    name_en: str
    product: str
    # Primary pages that usually announce the newest flagship model
    # along with benchmark numbers. Pages are tried in order; the
    # extractor is given the concatenated content of successful fetches.
    urls: List[str] = field(default_factory=list)
    # Hint for the LLM extractor — describes what the vendor calls its
    # flagship line so it can disambiguate when the page lists many models.
    flagship_hint: str = ""
    # ------------------------------------------------------------------
    # AA-first discovery (preferred): pick this vendor's flagship directly
    # from the Artificial Analysis /models list, using intelligence_index
    # as the ranking signal. Falls back to LLM discovery only if no AA
    # row survives the filter.
    # ------------------------------------------------------------------
    # AA row name must start with this (case-insensitive). e.g. "GPT-",
    # "Claude", "GLM-", "MiMo".
    aa_prefix: str = ""
    # Regex patterns (case-insensitive) — if any match the AA row name,
    # reject that row. Used to filter out cheaper siblings (nano/mini/
    # flash/haiku), domain-specific variants (codex/vl/audio/ocr), and
    # experimental builds (exp/preview-only) before ranking.
    aa_exclude_patterns: List[str] = field(default_factory=list)


VENDORS: List[Vendor] = [
    Vendor(
        id="openai",
        name_zh="OpenAI",
        name_en="OpenAI",
        product="ChatGPT",
        urls=[
            "https://openai.com/news/",
            "https://platform.openai.com/docs/models",
        ],
        flagship_hint="Latest GPT flagship (e.g. GPT-5, GPT-4.5, o-series reasoning).",
        aa_prefix="GPT-",
        # Drop size siblings (nano/mini), coding-specific variants (Codex),
        # and the non-reasoning ChatGPT consumer endpoint.
        aa_exclude_patterns=[
            r"\bnano\b",
            r"\bmini\b",
            r"\bcodex\b",
            r"\(ChatGPT\)",
        ],
    ),
    Vendor(
        id="anthropic",
        name_zh="Anthropic",
        name_en="Anthropic",
        product="Claude",
        urls=[
            "https://www.anthropic.com/news",
            "https://docs.anthropic.com/en/docs/about-claude/models/overview",
        ],
        flagship_hint="Latest Claude flagship model (Opus / Sonnet tier).",
        aa_prefix="Claude",
        # Only Opus counts as flagship; drop Sonnet / Haiku tiers.
        aa_exclude_patterns=[
            r"\bSonnet\b",
            r"\bHaiku\b",
        ],
    ),
    Vendor(
        id="google",
        name_zh="Google",
        name_en="Google",
        product="Gemini",
        urls=[
            "https://deepmind.google/models/gemini/",
            "https://blog.google/technology/google-deepmind/",
            "https://ai.google.dev/gemini-api/docs/models",
        ],
        flagship_hint=(
            "Latest Gemini flagship (Pro / Ultra tier). "
            "Include preview releases if they are the newest (e.g. 'Gemini 3.1 Pro Preview'). "
            "Always include the version number in display_name."
        ),
        aa_prefix="Gemini",
        # Drop all sub-flagship size tiers.
        aa_exclude_patterns=[
            r"\bFlash\b",
            r"\bNano\b",
            r"\bLite\b",
        ],
    ),
    Vendor(
        id="deepseek",
        name_zh="DeepSeek",
        name_en="DeepSeek",
        product="DeepSeek Chat",
        urls=[
            "https://api-docs.deepseek.com/news/",
            "https://api-docs.deepseek.com/quick_start/pricing",
            "https://github.com/deepseek-ai",
        ],
        flagship_hint=(
            "Latest DeepSeek flagship model. Prefer the newest V-series "
            "release (e.g. DeepSeek V3.2 / V3.1) over the older R1 reasoning "
            "model — R1 is from early 2025 and has been superseded. "
            "Always include the version suffix in display_name "
            "(e.g. 'DeepSeek V3.2', never just 'DeepSeek')."
        ),
        aa_prefix="DeepSeek",
        # Drop experimental builds and the "Speciale" research variant;
        # distilled siblings are not the flagship.
        aa_exclude_patterns=[
            r"\bExp\b",
            r"\bSpeciale\b",
            r"\bDistill",
            r"\bCoder\b",
        ],
    ),
    Vendor(
        id="zhipu",
        name_zh="智谱 AI",
        name_en="Zhipu AI",
        product="GLM",
        urls=[
            "https://open.bigmodel.cn/dev/howuse/model",
            "https://github.com/THUDM",
        ],
        flagship_hint="Latest GLM flagship (GLM-5 / GLM-4.5 tier).",
        aa_prefix="GLM-",
        # Exclude vision ("GLM-4.5V"), lightweight ("Air"), turbo/flash
        # variants, and modality-specific sibs (OCR/TTS/ASR/Image/Video).
        aa_exclude_patterns=[
            r"GLM-[\d.]+V\b",
            r"\bAir\b",
            r"\bTurbo\b",
            r"\bFlash\b",
            r"\bOCR\b",
            r"\bTTS\b",
            r"\bASR\b",
            r"\bImage\b",
            r"\bVideo\b",
        ],
    ),
    Vendor(
        id="xiaomi",
        name_zh="小米",
        name_en="Xiaomi",
        product="MiMo",
        urls=[
            "https://github.com/XiaomiMiMo",
            "https://github.com/XiaomiMiMo/MiMo-V2-Flash",
        ],
        flagship_hint=(
            "Latest Xiaomi MiMo flagship model. "
            "Prefer MiMo-V2-Pro > MiMo-V2-Omni > MiMo-V2-Flash > MiMo-V1. "
            "Always include the version suffix in display_name (e.g. 'MiMo-V2-Pro'), never just 'MiMo'."
        ),
        aa_prefix="MiMo",
        # Flash is mid-tier, VL/Audio/Embodied are modality sibs — strip
        # them all so the Pro (closed API flagship) wins on intelligence.
        aa_exclude_patterns=[
            r"\bFlash\b",
            r"\bVL\b",
            r"\bAudio\b",
            r"\bEmbodied\b",
        ],
    ),
    Vendor(
        id="minimax",
        name_zh="MiniMax",
        name_en="MiniMax",
        product="abab / MiniMax-Text",
        urls=[
            "https://www.minimaxi.com/news",
            "https://platform.minimaxi.com/document/ChatCompletion%20v2",
        ],
        flagship_hint="Latest MiniMax flagship (abab / MiniMax-Text / M-series).",
        aa_prefix="MiniMax",
        aa_exclude_patterns=[],
    ),
]


@dataclass
class Leaderboard:
    id: str
    name: str
    url: str
    # Field we will ask the extractor to populate for each vendor model
    score_field: str


# Note: primary per-benchmark scores and the Artificial Analysis
# Intelligence Index are pulled directly from AA's /models page by
# scripts/aa_parser.py (no LLM, no per-benchmark fetch). These
# LEADERBOARDS entries are the remaining third-party sources that
# still need HTML fetching + LLM extraction.
LEADERBOARDS: List[Leaderboard] = [
    Leaderboard(
        id="lmarena",
        name="LMArena",
        url="https://lmarena.ai/leaderboard",
        score_field="lmarena_elo",
    ),
    Leaderboard(
        id="livebench",
        name="LiveBench",
        url="https://livebench.ai/",
        score_field="livebench_avg",
    ),
]
