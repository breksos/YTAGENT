"""End-to-end agent loop smoke test with a scripted fake model — no network, no API keys.

Proves the LangChain wiring: create_agent runs, the fake model's tool call is
executed against a real workspace file, and the final answer surfaces via ask().
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import ytagent.agent as agent_mod
from ytagent.agent import Harness


class FakeToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self  # scripted responses; tool schemas are irrelevant


def test_agent_runs_tool_and_answers(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "transcript_abc.txt").write_text("the moon landing was in 1969\n", encoding="utf-8")

    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "grep_file", "args": {"pattern": "moon", "file": "transcript_abc.txt"}, "id": "c1"}],
            ),
            AIMessage(content="The transcript mentions the moon landing (line 1)."),
        ]
    )
    monkeypatch.setattr(agent_mod, "make_model", lambda m, t: FakeToolCallingModel(messages=script))
    monkeypatch.setattr(agent_mod, "find_yta", lambda: "yta-stub")

    steps = []
    harness = Harness(model_string="fake:model", workspace=ws)
    answer = harness.ask("what does the transcript say?", on_step=lambda k, t: steps.append((k, t)))

    assert answer == "The transcript mentions the moon landing (line 1)."
    assert ("tool", "grep_file(pattern='moon', file='transcript_abc.txt')") in steps
    assert any(k == "result" and "moon landing" in t for k, t in steps)
