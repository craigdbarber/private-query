"""Tests ollama_util."""

from collections.abc import Generator
from dataclasses import dataclass

import pytest
from test_util import start_local_ollama

from config_util import load_yaml_config
from ollama_util import OllamaClient


@dataclass
class _SessionData:
    client: OllamaClient
    model: str


@pytest.fixture(scope="session", name="session_data")
def suite_setup_teardown(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_SessionData, None, None]:
    """Set up ollama local service for testing."""
    # suite setup
    host = "localhost:11434"
    model = "llama3.2:latest"
    client: OllamaClient
    ollama_home_dir = tmp_path_factory.mktemp("ollama_home")
    start_local_ollama(host, ollama_home_dir)

    tmp_config_dir = tmp_path_factory.mktemp("config")
    tmp_config = tmp_config_dir / "config.yaml"
    tmp_config.write_text(f"""
host: "{host}"
model: "{model}"
                        """)
    config = load_yaml_config(str(tmp_config.absolute()))
    client = OllamaClient(config)

    yield _SessionData(client=client, model=model)


def test_ollama_client(session_data: _SessionData):
    """Test the OllamaClient is setup properly."""
    client = session_data.client
    assert client
    model = session_data.model
    assert model
    assert client.contains_model(model)


def test_execute_prompt(session_data: _SessionData):
    """Test execute_prompt."""
    client = session_data.client
    response = client.execute_prompt("Hello how are you?")
    assert response
