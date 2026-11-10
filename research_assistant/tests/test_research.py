import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant import (
    LLMError, MockFetcher, MockSearch, PipelineError, ResearchConfig, ResearchPipeline, make_llm,
)
from research_assistant.agents import ReportAgent, SearchAgent, clean_citations
from research_assistant.llm import LLM, MockLLM, parse_json
from research_assistant.models import SearchHit, Source, SourceSummary, Synthesis
from research_assistant.tools import SearchBackend, extract_text, is_public_url


class ScriptedLLM(LLM):
    """LLM whose replies come from a dict {role: str | callable}."""

    name = "mock"

    def __init__(self, replies):
        super().__init__("scripted")
        self.replies, self.calls = replies, []

    def complete(self, system, user, *, temperature=0.3, json_mode=False, max_tokens=None, role=""):
        self.calls.append(role)
        r = self.replies[role]
        return r(user) if callable(r) else r


# ---- parsing -----------------------------------------------------------------
def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"a": 1} Hope it helps.') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json("not json at all")


def test_clean_citations():
    text, bad = clean_citations("Claim [1] and [9] here [2].", {1, 2})
    assert "[9]" not in text and "[1]" in text and "[2]" in text and bad == {9}


# ---- tools -------------------------------------------------------------------
def test_extract_text_drops_scripts_and_chrome():
    html = ("<html><body><nav>MENU</nav><script>evil()</script>"
            "<article><h1>Title</h1><p>Real  content.</p></article><footer>F</footer></body></html>")
    out = extract_text(html)
    assert "Real content." in out and "Title" in out
    assert "evil" not in out and "MENU" not in out


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/admin", "http://localhost:8080", "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.5/", "http://192.168.1.1/", "file:///etc/passwd", "ftp://example.com", "not a url",
])
def test_ssrf_guard_blocks(url):
    assert is_public_url(url) is False


# ---- search agent ------------------------------------------------------------
def test_search_agent_dedupes_and_round_robins():
    class Backend(SearchBackend):
        def search(self, q, n=5):
            return [SearchHit("Shared", "https://a.com/x#frag"), SearchHit(f"{q}-1", f"https://b.com/{q}")]

    llm = ScriptedLLM({"planner": json.dumps({"queries": ["q1", "q2"]})})
    agent = SearchAgent(llm, Backend(), MockFetcher(), 0.3)
    queries, sources = agent.run("t", ResearchConfig(n_queries=2, max_sources=10), lambda m: None)
    assert queries == ["q1", "q2"]
    urls = [s.url for s in sources]
    assert len(urls) == len(set(u.split("#")[0] for u in urls))  # shared URL counted once
    assert [s.id for s in sources] == [1, 2, 3]


def test_planner_falls_back_to_topic_on_bad_json():
    llm = ScriptedLLM({"planner": "garbage"})
    agent = SearchAgent(llm, MockSearch(), MockFetcher(), 0.3)
    logs = []
    assert agent.plan_queries("my topic", 3, logs.append) == ["my topic"]
    assert llm.calls == ["planner", "planner"]  # retried once
    assert any("falling back" in m for m in logs)


def test_failed_fetch_falls_back_to_snippet():
    # MockFetcher raises for URLs containing "/3"
    agent = SearchAgent(MockLLM(), MockSearch(), MockFetcher(), 0.3)
    _, sources = agent.run("topic", ResearchConfig(n_queries=1, results_per_query=5, max_sources=5),
                           lambda m: None)
    assert any(not s.fetched for s in sources) and any(s.fetched for s in sources)


# ---- reporter ----------------------------------------------------------------
def test_reporter_strips_invented_citations():
    sources = [Source(1, "A", "https://a.com"), Source(2, "B", "https://b.com")]
    summaries = [SourceSummary(1, "s1"), SourceSummary(2, "s2")]
    bad_report = json.dumps({
        "title": "T", "executive_summary": "Summary [1] [7].",
        "key_findings": [{"text": "F [8]", "source_ids": [1, 99]}],
        "sections": [{"heading": "H", "body": "Body [2][5]."}],
        "limitations": [], "conclusion": "C [1].", "follow_up_questions": [],
    })
    agent = ReportAgent(ScriptedLLM({"reporter": bad_report}), 0.4)
    report = agent.run("t", summaries, Synthesis(), sources, lambda m: None)
    md = report.to_markdown()
    for bad in ("[7]", "[8]", "[5]", "[99]"):
        assert bad not in md
    assert report.key_findings[0].source_ids == [1]
    assert report.warnings and "non-existent" in report.warnings[0]


def test_report_markdown_structure():
    pipe = ResearchPipeline(MockLLM(), MockSearch(), MockFetcher())
    md = pipe.run("topic", ResearchConfig(n_queries=2, max_sources=4)).report.to_markdown()
    for heading in ("# ", "## Executive summary", "## Key findings", "## Conclusion", "## Sources"):
        assert heading in md


# ---- pipeline ----------------------------------------------------------------
def test_pipeline_event_order_and_result():
    pipe = ResearchPipeline(MockLLM(), MockSearch(), MockFetcher())
    events = list(pipe.run_iter("topic", ResearchConfig(n_queries=2, max_sources=4)))
    done_agents = [e.agent for e in events if e.stage == "done"]
    assert done_agents == ["Search", "Summarizer", "Reporter"]
    assert events[-1].stage == "result" and events[-1].data.report.title


def test_pipeline_uses_configured_temperatures():
    llm = MockLLM()
    cfg = ResearchConfig(n_queries=1, max_sources=2, report_temperature=0.9)
    ResearchPipeline(llm, MockSearch(), MockFetcher()).run("t", cfg)
    temps = {c["role"]: c["temperature"] for c in llm.calls}
    assert temps["reporter"] == 0.9 and temps["summarizer"] == cfg.summarizer_temperature


def test_irrelevant_sources_are_dropped():
    low = json.dumps({"summary": "off topic", "key_points": [], "relevance": 1, "reliability_notes": ""})
    llm = ScriptedLLM({
        "planner": json.dumps({"queries": ["q"]}),
        "summarizer": lambda user: low if "/shared" in user else json.dumps(
            {"summary": "ok", "key_points": ["x"], "relevance": 5, "reliability_notes": ""}),
        "synthesis": json.dumps({"themes": [], "agreements": [], "disagreements": [], "gaps": []}),
        "reporter": MockLLM().complete("", "[2]\n", role="reporter"),
    })
    result = ResearchPipeline(llm, MockSearch(), MockFetcher()).run(
        "t", ResearchConfig(n_queries=1, results_per_query=3, max_sources=3))
    assert all("shared" not in s.url for s in result.report.sources)


def test_no_sources_is_a_clean_error():
    class Empty(SearchBackend):
        def search(self, q, n=5):
            return []

    pipe = ResearchPipeline(MockLLM(), Empty(), MockFetcher())
    with pytest.raises(PipelineError, match="no sources"):
        pipe.run("t")


def test_search_backend_failure_is_reported_not_crashed():
    class Boom(SearchBackend):
        def search(self, q, n=5):
            raise RuntimeError("rate limited")

    events = list(ResearchPipeline(MockLLM(), Boom(), MockFetcher()).run_iter("t"))
    assert events[-1].stage == "error"
    assert any("rate limited" in e.message for e in events)


def test_make_llm_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError):
        make_llm("openai")
    with pytest.raises(LLMError):
        make_llm("gemini")
    with pytest.raises(LLMError):
        make_llm("bogus")
