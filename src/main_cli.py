"""A command line application for utilizing the private-query system."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import BaseModel

from chroma_util import ChromaClient, ChromaClientConfig
from config_util import load_yaml_config
from ollama_util import OllamaClient, OllamaClientConfig
from private_query import PrivateQuery
from resource_util import resolve_resource

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
    private_query: PrivateQuery
    collection_name: str


class _Settings(BaseModel):
    collection_name: str
    chroma: ChromaClientConfig
    ollama: OllamaClientConfig


def _initialize_session(config_path: Path) -> _SessionData:
    config = load_yaml_config(str(config_path))
    settings = _Settings.model_validate(config["settings"])
    collection_name = settings.collection_name
    logger.info("[SESSION INIT]: Loading ChromaDB client...")
    chroma = ChromaClient(settings.chroma)
    logger.info("[SESSION INIT]: Loading Ollama client...")
    ollama = OllamaClient(settings.ollama)
    logger.info("[SESSION INIT]: Loading Private Query...")
    private_query = PrivateQuery(chroma, ollama)
    return _SessionData(
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

    logger.remove(0)
    if debug_log:
        logger.add(sys.stderr, level="DEBUG")
    elif info_log:
        logger.add(sys.stderr, level="INFO")
    else:
        logger.add(sys.stderr, level="WARNING")

    resolved_config_path: Path
    if config is not None:
        resolved_config_path = resolve_resource(config)
    else:
        resolved_config_path = resolve_resource(_DEFAULT_CONFIG_FILE)

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
    resolved_resource = resolve_resource(path)

    if not resolved_resource.exists():
        logger.error(f"Resource: '{path}' does not exist.")
        raise typer.Exit(code=1)
    files: list[Path] = []
    if resolved_resource.is_dir():
        files = [f for f in resolved_resource.iterdir() if f.is_file()]
    else:
        files.append(resolved_resource.resolve())
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
