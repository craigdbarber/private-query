"""Provides utility functionality for loading configuration data."""

import io
import types
import typing
from pathlib import Path
from typing import Any, TypeGuard, get_args, get_origin
from typing import Any as TypingAny

import yaml
from loguru import logger

from transformer_util import extract_text_from_file


def get_config_value[T](config: dict[str, Any], name: str, expected_type: type[T]) -> T:
    """Return the requested value from the config. Performs type checking
    of the retrieved value against the expected type.

    Args:
        config: The dictionary the value will be retrieved from.
        name: The name of the value to be retrieved.
        expected_type: The expected type of the value.

    Returns: The type checked requested value.

    Raises:
        KeyError: If the specified name could not be found in the config.
        TypeError: If the type of the value does not match expected_type.

    """
    try:
        value = _get_config_value(config, name, expected_type)
    except KeyError as ke:
        logger.error(f"Failed to retrieve config value, missing name: {name}")
        raise ke
    return value


def get_config_value_or_none[T](
    config: dict[str, Any], name: str, expected_type: type[T]
) -> T | None:
    """Return the requested value from the config if exists, or else none.

    Args:
        config: The dictionary the value will be retrieved from.
        name: The name of the value to be retrieved.
        expected_type: The expected type of the value.

    Returns: The type checked requested value.

    Raises:
        TypeError: If the type of the value does not match expected_type.

    """
    value: T | None
    try:
        value = _get_config_value(config, name, expected_type)
    except KeyError:
        value = None
    return value


def load_yaml_config(file_path: str | Path) -> dict[str, Any]:
    """Read the yaml file at the specified path and parses it into a dictionary.

    Args:
        file_path: The path of the yaml file to be loaded.
    Returns: The dictionary parsed from the specified YAML file.

    """
    file_path = Path(file_path).resolve()
    return yaml.safe_load(io.StringIO(extract_text_from_file(file_path)))


def _get_config_value[T](
    config: dict[str, Any], name: str, expected_type: type[T]
) -> T:
    value = config[name]
    if not _is_expected_type(value, expected_type):
        logger.error(f"Failed to retrieve config value, unexpected type: {type(value)}")
        raise TypeError(f"Expected type: {type(T)} got: {type(value)}")
    return value


def _is_expected_type[T](value: Any, expected_type: type[T]) -> TypeGuard[T]:
    if expected_type is TypingAny:
        return True

    origin = get_origin(expected_type) or expected_type
    # handle unions
    if origin in (types.UnionType, typing.Union):
        return any(_is_expected_type(value, arg) for arg in get_args(expected_type))

    # standard runtime check
    try:
        return isinstance(value, origin)
    except TypeError:
        # fallback for unexpected edge case
        return False
