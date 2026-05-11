"""Provides utility functionality for loading configuration data."""

from typing import TypeAlias, Union, cast

import yaml

NestedDict: TypeAlias = dict[str, Union[str, "NestedDict"]]


def get_config_dict(
    config: NestedDict, name: str, raise_error: bool = True
) -> NestedDict | None:
    """Attempt to retrieve the configuration dictionary value with the specified name.

    Args:
        config: The configuration dict to retrieve the value from.
        name: The name of the value to be retrieved.
        raise_error: If an error should be raised when the requested value is not found.
    Returns: The retrieved value.

    Raises:
        ValueError: If the requested value is missing.

    """
    value = _get_config_value(config, name, raise_error)
    if value is None:
        return None
    return cast(NestedDict, value)


def get_config_str(
    config: NestedDict, name: str, raise_error: bool = True
) -> str | None:
    """Attempt to retrieve the configuration value with the specified name.
    If the specified value is not found and raise_error is True, will raise ValueError.

    Args:
            config: The configuration dictionary to retrieve the value from.
            name: The name of the value to be retrieved.
            raise_error: If this function should raise an error if the requested
            value is missing.
    Returns: The retrieved configuration value.

    Raises:
            ValueError: If the requested value is missing.

    """
    value = _get_config_value(config, name, raise_error)
    if value is None:
        return None
    return cast(str, value)


def load_yaml_config(file_path: str) -> NestedDict:
    """Read the yaml file at the specified path and parses it into a dictionary.

    Args:
        file_path: The path of the yaml file to be loaded.
    Returns: The dictionary parsed from the specified YAML file.

    Raises:
        FileNotFoundError: If the specified file path doesn't exists.

    """
    with open(file=file_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _get_config_value(
    config: NestedDict, name: str, raise_error: bool = True
) -> str | NestedDict | None:
    value = config.get(name)
    if value is not None:
        return value
    if raise_error:
        raise ValueError(f"Missing config value: {name}")
    return None
