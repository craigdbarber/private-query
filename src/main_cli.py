"""A command line application for utilizing the private-query system."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from chroma_util import ChromaClient
from config_util import get_config_value, load_yaml_config
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


def _initialize_session(config_path: Path) -> _SessionData:
    config = load_yaml_config(str(config_path))
    settings = get_config_value(config, "settings", dict[str, Any])
    collection_name = get_config_value(settings, "collection_name", str)
    logger.info("[SESSION INIT]: Loading ChromaDB client...")
    chroma_config = get_config_value(settings, "chroma", dict[str, str])
    chroma = ChromaClient(chroma_config)
    logger.info("[SESSION INIT]: Loading Ollama client...")
    ollama_config = get_config_value(settings, "ollama", dict[str, str])
    ollama = OllamaClient(ollama_config)
    logger.info("[SESSION INIT]: Loading Private Query...")
    private_query = PrivateQuery(chroma, ollama)
    return _SessionData(
        chroma=chroma,
        ollama=ollama,
        private_query=private_query,
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
    info_log: Annotated[
        bool, typer.Option("--info", help="Enable INFO logging.")
    ] = False,
    debug_log: Annotated[
        bool, typer.Option("--debug", help="Enabled DEBUG logging.")
    ] = False,
):
    """Global setup context."""
    if ctx.invoked_subcommand == "help" or "--help" in sys.argv or "-h" in sys.argv:
        return

    resolved_config_path: Path
    if config is not None:
        resolved_config_path = config
    else:
        resolved_config_path = Path.cwd() / _DEFAULT_CONFIG_FILE

    logger.remove(0)
    if debug_log:
        logger.add(sys.stderr, level="DEBUG")
    elif info_log:
        logger.add(sys.stderr, level="INFO")
    else:
        logger.add(sys.stderr, level="WARNING")

    ctx.obj = _initialize_session(resolved_config_path)


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
    print("Thinking...\n")
    print(
        session_data.private_query.process_prompt(
            prompt=prompt, collection_name=session_data.collection_name
        )
    )


@app.command("help")
def show_help(ctx: typer.Context):
    """Show the help message."""
    parent = ctx.parent
    if parent is not None:
        typer.echo(parent.get_help())


if __name__ == "__main__":
    app()
