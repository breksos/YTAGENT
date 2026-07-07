"""Typed LangChain tools wrapping the yta CLI (subprocess) plus workspace tools.

Design mirrors the CLI's own agent contract:
- transcript/comments always go to files in a session workspace (never into
  the model's context wholesale); the model then greps/reads slices.
- yta's one-line stderr errors and exit codes are translated into actionable
  tool output instead of exceptions, so the model can change strategy.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

YTA_TIMEOUT = int(os.environ.get("YTAGENT_YTA_TIMEOUT", "300"))

# Reaction guidance per yta exit code (from the CLI's error taxonomy).
_EXIT_GUIDANCE = {
    2: "Bad argument or URL. Fix the argument and retry.",
    3: "Video is private/removed/members-only. Skip this video and pick another.",
    4: "Age-restricted; not retrievable anonymously. Skip this video.",
    5: "No transcript in that language. The message lists available languages — "
       "retry with a listed lang code, or fall back to yta_meta/yta_comments.",
    6: "YouTube is rate-limiting. Do NOT retry immediately. If this was an "
       "auto-translated transcript request, fetch the original language "
       "(omit lang) and translate the text yourself.",
    7: "Network problem. Retry once.",
}


def find_yta() -> str:
    """Locate the yta executable: YTA_BIN env var, PATH, then the sibling CLI repo venv."""
    env = os.environ.get("YTA_BIN")
    if env:
        if Path(env).exists():
            return env
        raise FileNotFoundError(f"YTA_BIN points to a missing file: {env}")
    on_path = shutil.which("yta")
    if on_path:
        return on_path
    exe = "yta.exe" if os.name == "nt" else "yta"
    # __file__ = <root>/src/ytagent/yta_tools.py -> parents[2] is the project root
    sibling = Path(__file__).resolve().parents[2].parent / "YTAgent" / ".venv" / "Scripts" / exe
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError(
        "yta CLI not found. Install it (pip install -e <YTAgent repo>) so `yta` is on "
        "PATH, or set YTA_BIN to the executable."
    )


def _run_yta(yta_bin: str, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [yta_bin, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=YTA_TIMEOUT,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _error_text(code: int, stderr: str) -> str:
    guidance = _EXIT_GUIDANCE.get(code, "Unclassified error; read the message.")
    line = stderr or f"yta exited with code {code} and no message"
    return f"{line}\nWhat to do: {guidance}"


def _safe_name(fragment: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", fragment)[:60] or "item"


def _video_key(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return _safe_name(url)


def build_tools(workspace: Path, yta_bin: str) -> list:
    """Create the tool set bound to a session workspace directory."""
    workspace.mkdir(parents=True, exist_ok=True)

    def run(args: list[str]) -> tuple[int, str, str]:
        try:
            return _run_yta(yta_bin, args)
        except subprocess.TimeoutExpired:
            return 7, "", f"error[network]: yta timed out after {YTA_TIMEOUT}s"

    @tool
    def yta_search(query: str, limit: int = 10, full: bool = False) -> str:
        """Search YouTube. Returns one line per result: 'N. title | channel | views | duration | date | url'.

        Fast/flat by default: upload date shows '-' and view counts are rounded.
        Set full=True (slower, one extra request per result) when you need exact
        dates/views for every result; for just a few videos prefer yta_meta on them.
        Note: flat titles may be auto-translated to this machine's locale; yta_meta shows the original.
        """
        code, out, err = run(["search", query, "-n", str(limit)] + (["--full"] if full else []))
        if code != 0:
            return _error_text(code, err)
        return out or "(no results)"

    @tool
    def yta_channel(channel: str, limit: int = 30, full: bool = False) -> str:
        """List a channel's most recent videos, strictly newest first (position 1 is the latest upload).

        `channel` is a channel URL, @handle, or bare handle. Flat/fast by default
        (dates '-', rounded views); full=True adds exact upload dates and views.
        Guessed @handles are often wrong/squatted — if you get 0 videos or the wrong
        creator, find the real channel via yta_search then yta_meta's channel_url.
        """
        code, out, err = run(["channel", channel, "-n", str(limit)] + (["--full"] if full else []))
        if code != 0:
            return _error_text(code, err)
        note = f"\n[note] {err}" if err else ""
        return (out or "(no videos)") + note

    @tool
    def yta_meta(url: str) -> str:
        """Full metadata for one video: title, channel, channel_url, upload_date, exact views/likes, language, tags, description.

        `url` is a video URL or bare 11-char video ID. Use this to verify a
        candidate (date, views, original title) before fetching its transcript.
        """
        code, out, err = run(["meta", url])
        if code != 0:
            return _error_text(code, err)
        return out

    @tool
    def yta_transcript(url: str, lang: str = "", timestamps: bool = False) -> str:
        """Fetch a video transcript into a workspace FILE (not into this conversation), returning the file name, line count, and a short preview.

        Leave lang empty to get the video's original language (recommended —
        auto-translated tracks are usually blocked for anonymous access).
        timestamps=True prefixes each line with [MM:SS].
        After fetching, use grep_file to locate keywords, then read_lines around hits.
        """
        vid = _video_key(url)
        fname = f"transcript_{_safe_name(vid)}{'_ts' if timestamps else ''}{'_' + lang if lang else ''}.txt"
        fpath = workspace / fname
        args = ["transcript", url, "-o", str(fpath)]
        if lang:
            args += ["--lang", lang]
        if timestamps:
            args += ["--timestamps"]
        code, out, err = run(args)
        if code != 0:
            return _error_text(code, err)
        lines = fpath.read_text(encoding="utf-8").splitlines()
        preview = "\n".join(lines[:12])
        return (
            f"Saved to file: {fname} ({len(lines)} lines). {err}\n"
            f"Preview (first lines):\n{preview}\n"
            f"...\nUse grep_file(pattern, '{fname}') to search it, read_lines to read slices."
        )

    @tool
    def yta_comments(url: str, limit: int = 100, sort: str = "top") -> str:
        """Fetch video comments into a workspace FILE, returning the file name and a short preview.

        One comment per line: '[likes] author (date): text'; replies start with '> '.
        sort is 'top' (most liked) or 'new'. After fetching, grep_file / read_lines it.
        """
        vid = _video_key(url)
        fname = f"comments_{_safe_name(vid)}_{sort}.txt"
        fpath = workspace / fname
        code, out, err = run(["comments", url, "-n", str(limit), "--sort", sort, "-o", str(fpath)])
        if code != 0:
            return _error_text(code, err)
        lines = fpath.read_text(encoding="utf-8").splitlines()
        preview = "\n".join(lines[:8])
        return (
            f"Saved to file: {fname} ({len(lines)} lines). {err}\n"
            f"Preview (first lines):\n{preview}\n"
            f"...\nUse grep_file(pattern, '{fname}') to search it, read_lines to read slices."
        )

    def _resolve(file: str) -> Path:
        p = (workspace / file).resolve()
        if workspace.resolve() not in p.parents and p != workspace.resolve():
            raise ValueError("file must be inside the session workspace")
        return p

    @tool
    def grep_file(pattern: str, file: str, context: int = 0, max_matches: int = 40) -> str:
        """Search a workspace file with a case-insensitive regex; returns 'line_no: text' for each match.

        `file` is a file name previously returned by yta_transcript/yta_comments.
        context=N also shows N lines before/after each match. Prefer several
        specific greps over reading a whole file.
        """
        try:
            p = _resolve(file)
            rx = re.compile(pattern, re.IGNORECASE)
        except ValueError as e:
            return f"error: {e}"
        except re.error as e:
            return f"error: bad regex: {e}"
        if not p.exists():
            return f"error: no such file '{file}'. Files present: " + (
                ", ".join(f.name for f in workspace.iterdir()) or "(none)"
            )
        lines = p.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        hits = 0
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                hits += 1
                if hits > max_matches:
                    out.append(f"... more than {max_matches} matches; refine the pattern.")
                    break
                if context:
                    lo, hi = max(1, i - context), min(len(lines), i + context)
                    out.extend(f"{j}: {lines[j - 1]}" for j in range(lo, hi + 1))
                    out.append("--")
                else:
                    out.append(f"{i}: {line}")
        return "\n".join(out) if out else f"(no matches for /{pattern}/ in {file}, {len(lines)} lines)"

    @tool
    def read_lines(file: str, start: int = 1, end: int = 60) -> str:
        """Read lines [start, end] (1-indexed, max 300 at a time) from a workspace file."""
        try:
            p = _resolve(file)
        except ValueError as e:
            return f"error: {e}"
        if not p.exists():
            return f"error: no such file '{file}'"
        lines = p.read_text(encoding="utf-8").splitlines()
        start = max(1, start)
        end = min(len(lines), max(start, end), start + 299)
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"{file} lines {start}-{end} of {len(lines)}:\n{body}"

    @tool
    def list_workspace() -> str:
        """List the files fetched so far in this session's workspace."""
        files = sorted(workspace.iterdir())
        if not files:
            return "(workspace is empty — nothing fetched yet)"
        return "\n".join(
            f"{f.name} ({len(f.read_text(encoding='utf-8').splitlines())} lines)" for f in files
        )

    return [
        yta_search,
        yta_channel,
        yta_meta,
        yta_transcript,
        yta_comments,
        grep_file,
        read_lines,
        list_workspace,
    ]
