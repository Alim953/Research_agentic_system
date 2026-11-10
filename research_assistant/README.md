# Research Assistant (multi-agent)

Three cooperating agents turn a question into a cited, structured report, behind a Gradio UI.

```
Topic ──► Search agent ──► Summarizer agent ──► Report agent ──► Markdown report
          plan queries,     summarize each        structured JSON →
          search the web,   source, then compare  rendered report with
          fetch pages       sources               validated [n] citations
```

| Agent | What it does |
|-------|--------------|
| **Search** | Asks the LLM for diverse queries (background, recent developments, evidence, criticism), searches DuckDuckGo (no key needed), merges and de-duplicates results round-robin, and fetches page text concurrently. Falls back to the search snippet if a page can't be fetched. |
| **Summarizer** | Summarizes each source into key points with a relevance score (1-5) and reliability notes, then compares sources: themes, agreements, disagreements, gaps. Irrelevant sources are dropped. |
| **Reporter** | Writes an executive summary, key findings, sections, limitations, conclusion and follow-up questions as JSON, then renders Markdown. Citations to non-existent sources are stripped and flagged. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY and/or GEMINI_API_KEY
python -m research_assistant           # opens http://127.0.0.1:7860
```

Options: `--host`, `--port`, `--share` (public Gradio link). You can also paste an API key into
the UI instead of using `.env`; it is used for that run only and not stored.

**Try it with no keys:** choose provider `mock`. This runs the whole pipeline on fake search
results and a fake LLM, which is useful for seeing the interface.

### UI

- **Research depth**: quick (~4 sources), standard (~6), deep (~10).
- **Report temperature**: lower = more conservative wording. Planner (0.3) and summarizer (0.2)
  stay low so facts aren't embellished.
- **Agent activity** streams live; the Sources and Source summaries tabs fill in as each agent
  finishes, and the final report can be downloaded as Markdown.

## Use as a library

```python
from research_assistant import ResearchPipeline, ResearchConfig, make_llm, DuckDuckGoSearch, PageFetcher

pipe = ResearchPipeline(make_llm("gemini"), DuckDuckGoSearch(), PageFetcher())
result = pipe.run("Impact of remote work on city centers", ResearchConfig(max_sources=8))
print(result.report.to_markdown())

# or stream progress events:
for ev in pipe.run_iter("topic"):
    print(ev.agent, ev.stage, ev.message)
```

## Design notes

- **Untrusted web content.** Page text is passed to the model inside `<source_text>` tags with
  instructions to treat it as data. This reduces prompt-injection risk but cannot eliminate it,
  so review important reports.
- **Citations are checked.** The model may only cite sources it was given; invented `[n]`
  references are removed and reported in the activity log.
- **Fetcher safety.** Only http(s) URLs resolving to public IPs are fetched; redirects are
  re-checked, size and time are capped. (A determined DNS-rebinding attacker could still
  slip through; don't expose this to untrusted users without further hardening.)
- **Limits.** Reports reflect only what the fetched pages say. Paywalled, JavaScript-rendered or
  blocked pages fall back to snippets, and the Sources tab shows which ones.

## Tests

```bash
python -m pytest -q
```

The tests use the mock LLM/search, so they need no keys or network.

## Notes

- Default models are `gpt-4o-mini` and `gemini-2.5-flash`. Model names change; set
  `OPENAI_MODEL` / `GEMINI_MODEL` or use the Model field in the UI.
- Some newer OpenAI models only accept the default temperature; if you get a temperature
  error, pick a model that supports it.
- DuckDuckGo may rate-limit heavy use. Add another `SearchBackend` subclass (e.g. for Tavily,
  Brave or SerpAPI) in `tools.py` and pass it to `ResearchPipeline`.
