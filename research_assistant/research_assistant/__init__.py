"""Multi-agent research assistant: Search -> Summarizer -> Reporter."""

from .llm import LLMError, make_llm
from .models import DEPTH_PRESETS, ResearchConfig, ResearchResult
from .pipeline import PipelineError, ResearchPipeline
from .tools import DuckDuckGoSearch, MockFetcher, MockSearch, PageFetcher

__all__ = [
    "ResearchPipeline", "ResearchConfig", "ResearchResult", "DEPTH_PRESETS",
    "PipelineError", "LLMError", "make_llm",
    "DuckDuckGoSearch", "PageFetcher", "MockSearch", "MockFetcher",
]
__version__ = "1.0.0"
