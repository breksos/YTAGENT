"""ytagent CLI: one-shot question answering plus an interactive REPL.

    ytagent "what was X's latest video about?" --model ollama:llama3.1
    ytagent --model groq:llama-3.3-70b-versatile          # REPL (keeps history)
    ytagent models                                        # provider status

Model resolution: --model flag, else YTAGENT_MODEL env var.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import ConfigError, ENV_MODEL, provider_status, resolve_model_string

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Vendor-agnostic LangChain agent that answers questions using the yta YouTube CLI.",
)


def _step_printer(kind: str, text: str) -> None:
    prefix = {"tool": "->", "result": "  <-"}.get(kind, "  ..")
    print(f"{prefix} {text}", file=sys.stderr)


def _build_harness(model: Optional[str], workspace: Optional[Path], verbose: bool):
    from .agent import Harness, make_workspace  # deferred: heavy imports

    model_string = resolve_model_string(model)
    ws = make_workspace(workspace)
    print(f"model: {model_string}\nworkspace: {ws}", file=sys.stderr)
    return Harness(model_string=model_string, workspace=ws, verbose=verbose)


@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="Question to answer. Omit for an interactive REPL."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help=f"provider:model (e.g. ollama:llama3.1). Default: ${ENV_MODEL}."
    ),
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Directory for fetched files (default: fresh temp dir)."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress live tool-call progress on stderr."),
):
    """Answer a question about YouTube content (or start a REPL if no question given)."""
    on_step = None if quiet else _step_printer
    try:
        harness = _build_harness(model, workspace, verbose=not quiet)
    except (ConfigError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(2)

    if question:
        print(harness.ask(question, on_step=on_step))
        return

    print("ytagent REPL — ask about YouTube content. Ctrl+C or 'exit' to quit.", file=sys.stderr)
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        try:
            print("\n" + harness.ask(q, on_step=on_step))
        except KeyboardInterrupt:
            print("\n(interrupted)", file=sys.stderr)


@app.command()
def models():
    """Show known providers, whether their package is installed, and key status."""
    print(f"model selection: --model provider:model, or set {ENV_MODEL}\n")
    for provider, hint, installed, key_ok in provider_status():
        pkg = "installed" if installed else f"missing (run: {hint})"
        key = "key ok / not needed" if key_ok else "API key env var NOT set"
        print(f"{provider:14} {pkg:45} {key}")


def _version_callback(value: bool):
    if value:
        print(f"ytagent {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", "-V", callback=_version_callback, is_eager=True),
):
    pass


_COMMANDS = {"ask", "models"}
_PASSTHROUGH_FLAGS = {"-h", "--help", "-V", "--version"}


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    # UX sugar: `ytagent "question"` and bare `ytagent --model x` route to `ask`,
    # so users don't have to remember the subcommand.
    argv = sys.argv[1:]
    if not any(a in _PASSTHROUGH_FLAGS for a in argv):
        first_positional = next((a for a in argv if not a.startswith("-")), None)
        if first_positional not in _COMMANDS:
            argv = ["ask", *argv]
    app(args=argv)


if __name__ == "__main__":
    main()
