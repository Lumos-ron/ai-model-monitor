"""
Configuration for AI vendor sources and third-party leaderboards.

Each vendor entry lists one or more URLs whose HTML should be fetched
and handed to the LLM extractor. The extractor is told which vendor
it is looking at and returns structured JSON.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkPage:
    slug: str         # URL slug under /evaluations/
    label: str        # display name on the card (e.g. "GPQA Diamond")
    unit: str         # '%' for percentages, '' otherwise

    @property
    def url(self) -> str:
        return f"https://artificialanalysis.ai/evaluations/{self.slug}"


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
    ),
    Vendor(
        id="deepseek",
        name_zh="DeepSeek",
        name_en="DeepSeek",
        product="DeepSeek Chat",
        urls=[
            "https://api-docs.deepseek.com/news/",
            "https://github.com/deepseek-ai",
        ],
        flagship_hint="Latest DeepSeek flagship (DeepSeek-V3 / R1 reasoning line).",
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
        flagship_hint="Latest GLM flagship (GLM-4 / GLM-4.5 tier).",
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
            "Latest Xiaomi MiMo flagship open-source model. "
            "Prefer MiMo-V2-Pro > MiMo-V2-Omni > MiMo-V2-Flash > MiMo-V1. "
            "Always include the version suffix in display_name (e.g. 'MiMo-V2-Flash'), never just 'MiMo'."
        ),
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
    ),
]


@dataclass
class Leaderboard:
    id: str
    name: str
    url: str
    # Field we will ask the extractor to populate for each vendor model
    score_field: str


# Primary per-benchmark sources: Artificial Analysis publishes one
# leaderboard page per benchmark under /evaluations/{slug}, listing
# every model with its normalised score. Much more reliable than
# scraping each vendor's own announcement blog, where HTML shape
# varies wildly and score units are ambiguous.
AA_BENCHMARK_PAGES: List[BenchmarkPage] = [
    BenchmarkPage("mmlu-pro", "MMLU-Pro", "%"),
    BenchmarkPage("gpqa-diamond", "GPQA Diamond", "%"),
    BenchmarkPage("humanitys-last-exam", "Humanity's Last Exam", "%"),
    BenchmarkPage("livecodebench", "LiveCodeBench", "%"),
    BenchmarkPage("scicode", "SciCode", "%"),
    BenchmarkPage("math-500", "MATH-500", "%"),
    BenchmarkPage("aime-2025", "AIME 2025", "%"),
    BenchmarkPage("ifbench", "IFBench", "%"),
]


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
    Leaderboard(
        id="aa_intelligence_index",
        name="Artificial Analysis Intelligence Index",
        url="https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index",
        score_field="aa_intelligence_index",
    ),
]
