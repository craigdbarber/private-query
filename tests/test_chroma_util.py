"""Tests chroma_util."""

import tempfile
from pathlib import Path

from chroma_util import ChromaClient
from config_util import NestedDict, load_yaml_config


def test_client_setup():
    """Test ChromaClient is successfully setup."""
    with tempfile.TemporaryDirectory() as tmp_persist_dir:
        tmp_persist_dir_path = Path(tmp_persist_dir)
        with tempfile.NamedTemporaryFile(mode="w+t") as tmp_config:
            tmp_config.write(f"""
type: local
persist_directory: {tmp_persist_dir_path}
embedding_model: multi-qa-MiniLM-L6-cos-v1
                        """)
            tmp_config.flush()
            chroma_config: NestedDict = load_yaml_config(tmp_config.name)
            assert chroma_config is not None
            client: ChromaClient = ChromaClient(chroma_config)
            assert client is not None
