"""Gradio interface for the research assistant."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr

from .llm import LLMError, make_llm
from .models import DEPTH_PRESETS, ResearchConfig, ResearchResult
from .pipeline import ResearchPipeline
from .tools import DuckDuckGoSearch, MockFetcher, MockSearch, PageFetcher

SOURCE_HEADERS = ["#", "Title", "URL", "Fetched", "Relevance"]

INTRO = """# Research Assistant
Three agents work in sequence: **Search** plans queries and gathers pages, **Summarizer**
condenses each source and compares them, and **Reporter** writes a structured, cited report.
Use provider `mock` to try the interface offline without any API key."""


def build_pipeline(provider: str, model: str, api_key: str) -> ResearchPipeline:
    llm = make_llm(provider, model.strip() or None, api_key.strip() or None)
    if provider == "mock":
        return ResearchPipeline(llm, MockSearch(), MockFetcher())
    return ResearchPipeline(llm, DuckDuckGoSearch(), PageFetcher())


def build_config(depth: str, temperature: float) -> ResearchConfig:
    return ResearchConfig(**DEPTH_PRESETS[depth], report_temperature=float(temperature))


def _source_rows(sources, summaries=None) -> list[list]:
    rel = {s.source_id: s.relevance for s in (summaries or [])}
    return [
        [s.id, s.title, s.url, "yes" if s.fetched else "snippet only",
         f"{rel[s.id]}/5" if s.id in rel else ""]
        for s in sources
    ]


def _summaries_md(summaries, synthesis, by_id) -> str:
    parts = []
    for s in summaries:
        src = by_id[s.source_id]
        points = "\n".join(f"  - {p}" for p in s.key_points)
        parts.append(
            f"### [{s.source_id}] {src.title}\n{s.summary}\n\n{points}\n\n"
            f"*Relevance {s.relevance}/5. {s.reliability_notes}*"
        )
    if synthesis and any([synthesis.themes, synthesis.agreements,
                          synthesis.disagreements, synthesis.gaps]):
        def block(title, items):
            return f"**{title}**\n" + "\n".join(f"- {i}" for i in items) if items else ""
        parts.append(
            "### Cross-source analysis\n\n"
            + "\n\n".join(filter(None, [
                block("Themes", synthesis.themes),
                block("Where sources agree", synthesis.agreements),
                block("Where sources disagree", synthesis.disagreements),
                block("Gaps", synthesis.gaps),
            ]))
        )
    return "\n\n".join(parts)


def _save_report(result: ResearchResult) -> str:
    path = Path(tempfile.mkdtemp(prefix="research_")) / "research_report.md"
    path.write_text(result.report.to_markdown(), encoding="utf-8")
    return str(path)


def run_research(topic, provider, model, api_key, depth, temperature):
    """Generator handler: streams (log, report, summaries, sources table, file)."""
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Please enter a research topic.")
    try:
        pipeline = build_pipeline(provider, model or "", api_key or "")
    except LLMError as exc:
        raise gr.Error(str(exc)) from exc

    log: list[str] = []
    report_md, summaries_md, rows, file = "", "", [], None
    sources_by_id: dict = {}

    def snap():
        return "\n".join(log), report_md, summaries_md, rows, file

    yield snap()
    for ev in pipeline.run_iter(topic, build_config(depth, temperature)):
        log.append(f"[{ev.agent}] {ev.message}")
        if ev.stage == "error":
            gr.Warning(ev.message)
        elif ev.stage == "done" and ev.agent == "Search":
            sources_by_id = {s.id: s for s in ev.data}
            rows = _source_rows(ev.data)
        elif ev.stage == "done" and ev.agent == "Summarizer":
            summaries, synthesis = ev.data
            summaries_md = _summaries_md(summaries, synthesis, sources_by_id)
            rows = _source_rows(list(sources_by_id.values()), summaries)
        elif ev.stage == "done" and ev.agent == "Reporter":
            report_md = ev.data.to_markdown()
        elif ev.stage == "result":
            file = _save_report(ev.data)
        yield snap()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Research Assistant") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=2):
                topic = gr.Textbox(
                    label="Research topic or question", lines=2,
                    placeholder="e.g. What is the current state of solid-state battery technology?",
                )
                with gr.Row():
                    provider = gr.Dropdown(["openai", "gemini", "mock"], value="openai",
                                           label="LLM provider")
                    model = gr.Textbox(label="Model (optional)", placeholder="provider default")
                api_key = gr.Textbox(
                    label="API key (optional)", type="password",
                    placeholder="leave blank to use OPENAI_API_KEY / GEMINI_API_KEY from .env",
                )
                depth = gr.Radio(list(DEPTH_PRESETS), value="standard", label="Research depth",
                                 info="quick: ~4 sources, standard: ~6, deep: ~10")
                temperature = gr.Slider(0.0, 1.5, value=0.4, step=0.05,
                                        label="Report temperature",
                                        info="Lower = more conservative wording")
                go = gr.Button("Run research", variant="primary")
            with gr.Column(scale=1):
                log = gr.Textbox(label="Agent activity", lines=16, interactive=False)

        with gr.Tabs():
            with gr.Tab("Report"):
                report = gr.Markdown()
                download = gr.File(label="Download report (Markdown)")
            with gr.Tab("Source summaries"):
                summaries = gr.Markdown()
            with gr.Tab("Sources"):
                sources = gr.Dataframe(headers=SOURCE_HEADERS, interactive=False)

        go.click(
            run_research,
            inputs=[topic, provider, model, api_key, depth, temperature],
            outputs=[log, report, summaries, sources, download],
            api_name="research",
        )
        topic.submit(
            run_research,
            inputs=[topic, provider, model, api_key, depth, temperature],
            outputs=[log, report, summaries, sources, download],
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Launch the research assistant web UI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = ap.parse_args(argv)
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
