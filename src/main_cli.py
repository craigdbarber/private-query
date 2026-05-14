"""A command line application for utilizing the private-query system."""

import config_util
from chroma_util import ChromaClient
from ollama_util import OllamaClient
from private_query import PrivateQuery

_CONFIG_FILE_PATH = "config.yaml"
if __name__ == "__main__":
    config = config_util.load_yaml_config(_CONFIG_FILE_PATH)
    settings = config_util.get_config_dict(config, "settings")
    assert settings is not None
    chroma_config = config_util.get_config_dict(settings, "chroma")
    assert chroma_config is not None
    chroma = ChromaClient(chroma_config)
    ollama_config = config_util.get_config_dict(settings, "ollama")
    assert ollama_config is not None
    ollama = OllamaClient(ollama_config)
    private_query = PrivateQuery(chroma, ollama)
