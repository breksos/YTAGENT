"""Tool behavior tests — no network, no LLM. The yta subprocess layer is stubbed."""

from pathlib import Path

import pytest

import ytagent.yta_tools as yt
from ytagent.yta_tools import build_tools


def tools_by_name(workspace: Path) -> dict:
    return {t.name: t for t in build_tools(workspace, yta_bin="yta-stub")}


@pytest.fixture
def ws(tmp_path):
    return tmp_path / "ws"


def stub_run(responses):
    """Return a _run_yta replacement keyed by yta subcommand name."""

    def fake(yta_bin, args):
        handler = responses[args[0]]
        return handler(args) if callable(handler) else handler

    return fake


def test_search_success(ws, monkeypatch):
    monkeypatch.setattr(yt, "_run_yta", stub_run({"search": (0, "1. Video A | Chan | 100 views | 3:00 | - | https://u", "")}))
    out = tools_by_name(ws)["yta_search"].invoke({"query": "test", "limit": 5})
    assert out.startswith("1. Video A")


def test_error_maps_exit_code_to_guidance(ws, monkeypatch):
    stderr = "error[no-transcript]: no subs for 'xx' (hint: available: en, tr)"
    monkeypatch.setattr(yt, "_run_yta", stub_run({"transcript": (5, "", stderr)}))
    out = tools_by_name(ws)["yta_transcript"].invoke({"url": "dQw4w9WgXcQ"})
    assert "error[no-transcript]" in out
    assert "What to do:" in out
    assert "retry with a listed lang code" in out


def test_blocked_guidance_discourages_retry(ws, monkeypatch):
    monkeypatch.setattr(yt, "_run_yta", stub_run({"comments": (6, "", "error[blocked]: rate limited")}))
    out = tools_by_name(ws)["yta_comments"].invoke({"url": "dQw4w9WgXcQ"})
    assert "Do NOT retry immediately" in out


def test_transcript_writes_file_and_previews(ws, monkeypatch):
    body = "\n".join(f"line {i} of the talk" for i in range(1, 41))

    def transcript(args):
        out_file = Path(args[args.index("-o") + 1])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(body + "\n", encoding="utf-8")
        return 0, f"wrote {out_file}", "transcript: lang=en kind=auto lines=40"

    monkeypatch.setattr(yt, "_run_yta", stub_run({"transcript": transcript}))
    tools = tools_by_name(ws)
    out = tools["yta_transcript"].invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert "transcript_dQw4w9WgXcQ.txt" in out
    assert "(40 lines)" in out
    assert "lang=en kind=auto" in out          # stderr summary surfaced to the model
    assert "line 13" not in out                 # preview only, not the whole file

    # and the workspace tools can work with it
    grep = tools["grep_file"].invoke({"pattern": r"line 1\d ", "file": "transcript_dQw4w9WgXcQ.txt"})
    assert grep.splitlines()[0].startswith("10:")
    read = tools["read_lines"].invoke({"file": "transcript_dQw4w9WgXcQ.txt", "start": 39, "end": 45})
    assert "40: line 40" in read and "lines 39-40 of 40" in read
    listing = tools["list_workspace"].invoke({})
    assert "transcript_dQw4w9WgXcQ.txt (40 lines)" in listing


def test_grep_no_match_and_missing_file(ws):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "a.txt").write_text("hello\n", encoding="utf-8")
    tools = tools_by_name(ws)
    assert "(no matches" in tools["grep_file"].invoke({"pattern": "zzz", "file": "a.txt"})
    missing = tools["grep_file"].invoke({"pattern": "x", "file": "nope.txt"})
    assert "no such file" in missing and "a.txt" in missing


def test_workspace_escape_is_blocked(ws):
    ws.mkdir(parents=True, exist_ok=True)
    tools = tools_by_name(ws)
    out = tools["read_lines"].invoke({"file": "../../secrets.txt"})
    assert "error" in out and "workspace" in out


def test_bad_regex_is_reported_not_raised(ws):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "a.txt").write_text("hello\n", encoding="utf-8")
    out = tools_by_name(ws)["grep_file"].invoke({"pattern": "(", "file": "a.txt"})
    assert "bad regex" in out
