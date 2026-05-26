"""Provides utility functionality for loading configuration data."""

import io
from pathlib import Path
from typing import Any

import yaml

from transformer_util import extract_text_from_file


def load_yaml_config(file_path: str | Path) -> dict[str, Any]:
    """Read the yaml file at the specified path and parses it into a dictionary.

    Args:
        file_path: The path of the yaml file to be loaded.
    Returns: The dictionary parsed from the specified YAML file.

    """
    file_path = Path(file_path).resolve()
    return yaml.safe_load(io.StringIO(extract_text_from_file(file_path)))
