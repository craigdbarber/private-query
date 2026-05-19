"""Tests ollama_util."""

import tempfile
from collections.abc import Generator
from dataclasses import dataclass

import pytest
from test_util import start_local_ollama, stop_process

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
    ollama_home_dir = tempfile.TemporaryDirectory()
    start_local_ollama(host, ollama_home_dir)

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
    stop_process("ollama")
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
