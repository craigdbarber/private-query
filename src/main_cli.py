"""A command line application for utilizing the private-query system."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

import config_util
from chroma_util import ChromaClient
from ollama_util import OllamaClient
from private_query import PrivateQuery

_DEFAULT_CONFIG_FILE = "config.yaml"

# Initialize the application.
app = typer.Typer(
    help="private-query: A modern RAG-style CLI tool for embedding documents and \
    executing user prompts using embedded data as context.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dataclass
class _SessionData:
    chroma: ChromaClient
    ollama: OllamaClient
    private_query: PrivateQuery
    collection_name: str
    verbose: bool


def _log(message: str, verbose: bool):
    if verbose:
        typer.echo(message)


def _initialize_session(config_path: Path, verbose: bool) -> _SessionData:
    config = config_util.load_yaml_config(str(config_path))
    settings = config_util.get_config_dict(config, "settings")
    assert settings is not None
    collection_name = config_util.get_config_str(settings, "collection_name")
    assert collection_name is not None
    _log("[SESSION INIT]: Loading ChromaDB client...", verbose)
    chroma_config = config_util.get_config_dict(settings, "chroma")
    assert chroma_config is not None
    chroma = ChromaClient(chroma_config)
    _log("[SESSION INIT]: Loading Ollama client...", verbose)
    ollama_config = config_util.get_config_dict(settings, "ollama")
    assert ollama_config is not None
    ollama = OllamaClient(ollama_config)
    _log("[SESSION INIT]: Loading Private Query...", verbose)
    private_query = PrivateQuery(chroma, ollama)
    return _SessionData(
        chroma=chroma,
        ollama=ollama,
        private_query=private_query,
        verbose=verbose,
        collection_name=collection_name,
    )


@app.callback()
def global_options(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            envvar="PRIVATE_QUERY_CONFIG",
            help="Path to configuration file. Priorities: 1. Flag, 2. \
                $PRIVATE_QUERY_CONFIG, Default.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,  # workaround for typer bug with bool default value :S
):
    """Global setup context."""
    if ctx.invoked_subcommand == "help" or "--help" in sys.argv or "-h" in sys.argv:
        return

    resolved_config_path: Path
    if config is not None:
        resolved_config_path = config
    else:
        resolved_config_path = Path.cwd() / _DEFAULT_CONFIG_FILE

    ctx.obj = _initialize_session(resolved_config_path, verbose)


@app.command()
def load(
    ctx: typer.Context,
    path: Annotated[
        Path,
        typer.Argument(..., help="The file path or directory containing documents."),
    ],
):
    """Load and embed a document or an entire directory of documents."""
    if not path.exists():
        typer.echo(f"Error: Path '{path}' does not exist.", err=True)
        raise typer.Exit(code=1)
    files: list[Path] = []
    if path.is_dir():
        files = [
            Path(path / f).resolve()
            for f in os.listdir(path)
            if os.path.isfile(path / f)
        ]
    else:
        files.append(Path(path).resolve())
    session_data: _SessionData = ctx.obj
    log_msg = "Loading and embedding documents:\n\n"
    for file in files:
        log_msg += f"{file}\n\n"
    _log(log_msg, session_data.verbose)
    session_data.private_query.embed_documents(
        collection_name=session_data.collection_name, document_paths=files
    )


@app.command()
def query(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(..., help="The prompt to be queried.")],
):
    """Excute the specified prompt."""
    session_data: _SessionData = ctx.obj
    _log("Executing query...\n\n", session_data.verbose)
    print("\n\nThinking...\n\n")
    print(
        session_data.private_query.process_prompt(
            prompt=prompt, collection_name=session_data.collection_name
        )
    )


@app.command("help")
def show_help(ctx: typer.Context):
    """Show the help message."""
    parent = ctx.parent
    assert parent
    typer.echo(parent.get_help())


if __name__ == "__main__":
    app()
