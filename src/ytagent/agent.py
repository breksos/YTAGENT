"""Agent assembly — the only module that touches LangChain's agent API.

Everything else (tools, prompts, config) is framework-independent, so if
LangChain's agent surface churns again, or Phase 3 grows into a custom
LangGraph graph with filter-worker sub-agents, this is the one file to change.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from .config import make_model
from .prompts import SYSTEM_PROMPT
from .yta_tools import build_tools, find_yta

DEFAULT_RECURSION_LIMIT = 80  # ~40 model/tool round-trips per question


@dataclass
class Harness:
    """A configured agent session: `ask()` one question or many (REPL keeps history)."""

    model_string: str
    workspace: Path
    verbose: bool = False
    temperature: float = 0.0
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        yta_bin = find_yta()
        model = make_model(self.model_string, self.temperature)
        self._agent = create_agent(
            model,
            tools=build_tools(self.workspace, yta_bin),
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    def ask(self, question: str, on_step=None) -> str:
        """Run one question through the agent; returns the final answer text.

        `on_step(kind, text)` receives progress events ('tool' calls and
        intermediate 'model' notes) for live display.
        """
        config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        }
        final = ""
        for chunk in self._agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                for msg in (update or {}).get("messages", []):
                    final = _report(msg, node, on_step) or final
        return final or "(the agent produced no final answer)"


def _report(msg, node: str, on_step) -> str | None:
    """Feed progress events to on_step; return content if msg is a final AI answer."""
    msg_type = getattr(msg, "type", "")
    if msg_type == "ai":
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            if on_step:
                for call in calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
                    on_step("tool", f"{call.get('name')}({args})")
            return None
        text = _text_of(msg)
        return text or None
    if msg_type == "tool" and on_step:
        text = _text_of(msg)
        first = text.splitlines()[0] if text else ""
        on_step("result", first[:200])
    return None


def _text_of(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content blocks
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def make_workspace(base: Path | None = None) -> Path:
    """Create a fresh session workspace directory for fetched files."""
    if base:
        base.mkdir(parents=True, exist_ok=True)
        return base
    return Path(tempfile.mkdtemp(prefix="ytagent_"))
