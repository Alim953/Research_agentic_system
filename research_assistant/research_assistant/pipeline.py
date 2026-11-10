"""Orchestrates the three agents and streams progress events."""

from __future__ import annotations

import queue
import threading
from typing import Iterator

from .agents import ReportAgent, SearchAgent, SummarizerAgent
from .llm import LLM
from .models import ProgressEvent, ResearchConfig, ResearchResult
from .tools import PageFetcher, SearchBackend


class PipelineError(RuntimeError):
    pass


class ResearchPipeline:
    def __init__(self, llm: LLM, backend: SearchBackend, fetcher: PageFetcher):
        self.llm = llm
        self.backend = backend
        self.fetcher = fetcher

    # -- the actual work ---------------------------------------------------
    def _run(self, topic: str, cfg: ResearchConfig, emit) -> ResearchResult:
        emit("Pipeline", f"Researching: {topic}", "start")

        # 1. Search agent
        emit("Search", "Planning queries and searching the web...", "start")
        searcher = SearchAgent(self.llm, self.backend, self.fetcher, cfg.planner_temperature)
        queries, sources = searcher.run(topic, cfg, lambda m: emit("Search", m))
        if not sources:
            raise PipelineError(
                "The search agent found no sources. Check your internet connection or "
                "try a different topic."
            )
        emit("Search", f"Collected {len(sources)} sources.", "done", sources)

        # 2. Summarizer agent
        emit("Summarizer", "Summarizing each source...", "start")
        summarizer = SummarizerAgent(self.llm, cfg.summarizer_temperature)
        log = lambda m: emit("Summarizer", m)
        summaries = summarizer.run(topic, sources, log)
        if not summaries:
            raise PipelineError("None of the sources could be summarized (see the activity log).")
        # Drop sources the summarizer judged irrelevant, unless that would leave nothing.
        usable = [s for s in summaries if s.relevance >= 2] or summaries
        if len(usable) < len(summaries):
            log(f"Ignoring {len(summaries) - len(usable)} irrelevant source(s).")
        synthesis = summarizer.synthesize(topic, usable, log)
        emit("Summarizer", f"Summarized {len(usable)} sources.", "done", (usable, synthesis))

        # 3. Report agent
        emit("Reporter", "Writing the structured report...", "start")
        reporter = ReportAgent(self.llm, cfg.report_temperature)
        report = reporter.run(topic, usable, synthesis, sources, lambda m: emit("Reporter", m))
        emit("Reporter", "Report complete.", "done", report)

        return ResearchResult(topic, queries, sources, usable, synthesis, report)

    # -- streaming interface (used by the UI) --------------------------------
    def run_iter(self, topic: str, cfg: ResearchConfig | None = None) -> Iterator[ProgressEvent]:
        cfg = cfg or ResearchConfig()
        q: queue.Queue = queue.Queue()

        def emit(agent, message, stage="progress", data=None):
            q.put(ProgressEvent(agent, message, stage, data))

        def worker():
            try:
                result = self._run(topic, cfg, emit)
                emit("Pipeline", "Done.", "result", result)
            except Exception as exc:  # surface any failure as an event
                emit("Pipeline", str(exc) or type(exc).__name__, "error")
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while (event := q.get()) is not None:
            yield event

    # -- blocking interface (library use) -------------------------------------
    def run(self, topic: str, cfg: ResearchConfig | None = None) -> ResearchResult:
        for ev in self.run_iter(topic, cfg):
            if ev.stage == "error":
                raise PipelineError(ev.message)
            if ev.stage == "result":
                return ev.data
        raise PipelineError("Pipeline ended without a result.")
