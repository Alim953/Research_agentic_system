"""Plain data containers passed between the agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


@dataclass
class Source:
    id: int
    title: str
    url: str
    snippet: str = ""
    content: str = ""      # extracted page text (empty if the fetch failed)
    fetched: bool = False
    query: str = ""        # the search query that found it


@dataclass
class SourceSummary:
    source_id: int
    summary: str
    key_points: list[str] = field(default_factory=list)
    relevance: int = 3     # 1 (irrelevant) .. 5 (highly relevant)
    reliability_notes: str = ""


@dataclass
class Synthesis:
    themes: list[str] = field(default_factory=list)
    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass
class Finding:
    text: str
    source_ids: list[int] = field(default_factory=list)


@dataclass
class ReportSection:
    heading: str
    body: str


@dataclass
class Report:
    title: str
    executive_summary: str
    key_findings: list[Finding]
    sections: list[ReportSection]
    limitations: list[str]
    conclusion: str
    follow_up_questions: list[str]
    sources: list[Source]
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        out = [f"# {self.title}", "", "## Executive summary", "", self.executive_summary, ""]
        out += ["## Key findings", ""]
        for f in self.key_findings:
            cites = " " + "".join(f"[{i}]" for i in f.source_ids) if f.source_ids else ""
            out.append(f"- {f.text}{cites}")
        out.append("")
        for s in self.sections:
            out += [f"## {s.heading}", "", s.body, ""]
        if self.limitations:
            out += ["## Limitations and caveats", ""]
            out += [f"- {x}" for x in self.limitations] + [""]
        out += ["## Conclusion", "", self.conclusion, ""]
        if self.follow_up_questions:
            out += ["## Suggested follow-up questions", ""]
            out += [f"- {q}" for q in self.follow_up_questions] + [""]
        out += ["## Sources", ""]
        out += [f"{s.id}. [{s.title or s.url}]({s.url})" for s in self.sources]
        return "\n".join(out).strip() + "\n"


@dataclass
class ResearchConfig:
    n_queries: int = 3
    results_per_query: int = 5
    max_sources: int = 6
    planner_temperature: float = 0.3
    summarizer_temperature: float = 0.2
    report_temperature: float = 0.4


# Depth presets used by the UI
DEPTH_PRESETS = {
    "quick": dict(n_queries=2, results_per_query=4, max_sources=4),
    "standard": dict(n_queries=3, results_per_query=5, max_sources=6),
    "deep": dict(n_queries=4, results_per_query=5, max_sources=10),
}


@dataclass
class ResearchResult:
    topic: str
    queries: list[str]
    sources: list[Source]
    summaries: list[SourceSummary]
    synthesis: Synthesis
    report: Report


@dataclass
class ProgressEvent:
    agent: str                 # "Pipeline", "Search", "Summarizer", "Reporter"
    message: str
    stage: str = "progress"    # start | progress | done | error | result
    data: Any = None
