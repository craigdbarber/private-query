"""Tests ollama_util."""

from collections.abc import Generator
from dataclasses import dataclass

import pytest
from test_util import start_local_ollama

from config_util import load_yaml_config
from ollama_util import OllamaClient, OllamaClientConfig


@dataclass
class _OllamaSessionData:
    client: OllamaClient
    model: str


@pytest.fixture(scope="session", name="ollama_session_data")
def suite_setup_teardown(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_OllamaSessionData, None, None]:
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
    yaml_config = load_yaml_config(str(tmp_config.absolute()))
    config = OllamaClientConfig.model_validate(yaml_config)
    client = OllamaClient(config)

    yield _OllamaSessionData(client=client, model=model)


def test_ollama_client(ollama_session_data: _OllamaSessionData):
    """Test the OllamaClient is setup properly."""
    client = ollama_session_data.client
    assert client
    model = ollama_session_data.model
    assert model
    assert client.contains_model(model)


def test_execute_prompt(ollama_session_data: _OllamaSessionData):
    """Test execute_prompt."""
    client = ollama_session_data.client
    response = client.execute_prompt("Hello how are you?")
    assert response
