# YTHarness

The agent harness for [YTCLI](https://github.com/breksos/YTCLI): a **vendor-agnostic LangChain agent**
(`ytagent`) that operates the `yta` YouTube CLI as its hands — searching, checking
metadata, dumping transcripts/comments to files, grepping them, and answering
research questions with cited videos.

This is "Part 2" of the YTCLI README: the CLI knows nothing about LLMs; this
project knows nothing about YouTube scraping.

## Architecture

```
src/ytagent/
  config.py      # provider:model resolution + preflight (framework-independent)
  prompts.py     # the operating playbook (framework-independent)
  yta_tools.py   # typed tools wrapping yta via subprocess + workspace grep/read tools
  agent.py       # the ONLY file that touches LangChain's agent API (create_agent)
  cli.py         # `ytagent` entry point: one-shot + REPL + `ytagent models`
```

- Built on **LangChain 1.x `create_agent`**, which compiles to a LangGraph graph —
  when Phase 3 (multi-model filter workers) arrives, this agent composes into a
  larger graph without a rewrite. Only `agent.py` would change.
- **Transcripts and comments never enter the model context wholesale.** The tools
  save them to a session workspace directory and return a preview; the agent then
  uses `grep_file` / `read_lines` — the same write-then-grep discipline the CLI
  was designed around.
- yta's exit-code error taxonomy is translated into "What to do" guidance in tool
  output, so the model changes strategy instead of retrying blindly.

## Setup

Managed with [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`).
Python 3.12 is pinned via `.python-version`; `uv.lock` pins the full dependency
tree (important given LangChain's API-churn history).

```powershell
uv sync                               # venv + all pinned deps + dev tools
# then the provider(s) you want:
uv sync --extra ollama                # local, no key
uv sync --extra groq --extra google   # free-tier friendly
# or everything: uv sync --extra all-providers
```

Run the CLI with `uv run ytagent ...` (or activate `.venv` and call `ytagent`
directly).

`yta` must be reachable: on PATH, via the `YTA_BIN` env var, or (fallback) the
sibling CLI repo's venv (`..\YTCLI\.venv\Scripts\yta.exe`; `..\YTAgent` is also
checked for older checkouts).

## Choosing a model (vendor-agnostic)

The model is a `provider:model` string — a CLI flag or the `YTAGENT_MODEL` env var:

```powershell
ytagent "..." --model ollama:llama3.1
ytagent "..." --model groq:llama-3.3-70b-versatile      # needs GROQ_API_KEY
ytagent "..." --model google_genai:gemini-2.5-flash     # needs GOOGLE_API_KEY
ytagent "..." --model openai:gpt-4o-mini                # needs OPENAI_API_KEY
ytagent "..." --model anthropic:claude-sonnet-5         # needs ANTHROPIC_API_KEY
```

API keys use each provider's standard env var. `ytagent models` shows what's
installed and which keys are set.

## Usage

```powershell
# one-shot
ytagent "What was the latest video on @veritasium about, and what did commenters think?"

# interactive REPL (conversation history is kept within the session)
ytagent

# keep fetched transcripts/comments somewhere inspectable
ytagent "..." -w .\session1

# suppress live tool-call progress (printed to stderr)
ytagent "..." -q
```

The final answer goes to stdout; progress and diagnostics go to stderr, so
`ytagent "..." -q > answer.txt` works cleanly.

## Tests

```powershell
uv run pytest
```

Tests run fully offline: the yta subprocess layer is stubbed, and the agent-loop
smoke test uses a scripted fake chat model — no API keys, no network.

## Notes on model choice

- **Tool-calling quality matters more than raw smarts here.** Small local models
  (7–8B) frequently mangle multi-argument tool calls; if the agent loops or stalls
  on Ollama, try a larger local model (e.g. `qwen2.5:32b`) or a hosted free tier
  (Groq, Gemini Flash).
- Temperature is pinned to 0 for tool-use reliability.
- A run is capped at ~40 model/tool round-trips (`recursion_limit`) so a confused
  model can't loop forever.
