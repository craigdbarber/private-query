"""Tests ollama_util."""

import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

from config_util import load_yaml_config
from ollama_util import OllamaClient


@dataclass
class _SessionData:
    client: OllamaClient
    model: str


@pytest.fixture(scope="session", name="session_data")
def suite_setup_teardown() -> Generator[_SessionData, None, None]:
    """Set up ollama local service for testing."""
    # suite setup
    host = "localhost:8080"
    model = "llama3.2:latest"
    client: OllamaClient
    start_script_path: Path
    stop_script_path: Path
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    project_path = Path(__file__).parent.parent

    # if running in bazel
    if runfiles_dir:
        glob_result = Path(runfiles_dir).rglob("scripts/start_ollama.sh")
        start_script_path = glob_result.__next__()
        glob_result = Path(runfiles_dir).rglob("scripts/stop_ollama.sh")
        stop_script_path = glob_result.__next__()
    # if running pytest directly
    elif project_path:
        start_script_path = project_path / "scripts" / "start_ollama.sh"
        stop_script_path = project_path / "scripts" / "stop_ollama.sh"
    else:
        raise FileNotFoundError(
            "Could not find scripts/start_ollama.sh and scripts/stop_ollama.sh"
        )
    assert start_script_path.exists()
    assert stop_script_path.exists()
    ollama_home_dir = tempfile.TemporaryDirectory()
    subprocess.run(  # noqa: S603
        [
            str(start_script_path),
            "--host",
            shlex.quote(host),
            "--ollama_home_dir",
            shlex.quote(str(Path(ollama_home_dir.name))),
        ],
        check=True,
    )
    # sleep to ensure the ollama server has started.
    time.sleep(5)

    with tempfile.NamedTemporaryFile(mode="w+t") as tmp_config:
        tmp_config.write(f"""
host: "{host}"
model: "{model}"
                         """)
        tmp_config.flush()
        config = load_yaml_config(tmp_config.name)
        client = OllamaClient(config)

    yield _SessionData(client=client, model=model)

    # suite teardown
    subprocess.run([str(stop_script_path)], check=True)  # noqa: S603
    ollama_home_dir.cleanup()


def test_ollama_client(session_data: _SessionData):
    """Test the OllamaClient is setup properly."""
    client = session_data.client
    assert client is not None
    model = session_data.model
    assert model is not None and model != ""
    assert client.contains_model(model)


def test_execute_prompt(session_data: _SessionData):
    """Test execute_prompt."""
    client = session_data.client
    response = client.execute_prompt("Hello how are you?")
    assert response is not None and response != ""
    print(f"Response: {response}")
