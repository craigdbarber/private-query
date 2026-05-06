import yaml


def get_config_value(
    config: dict[str, dict[str, str] | str], name: str, raise_error: bool = True
) -> dict[str, str] | str | None:
    """Attempts to retrieve the configuration value with the specified name.
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
    value = config.get(name)
    if value is not None:
        return value
    if raise_error:
        raise ValueError(f"Missing config value: ${name}")
    return None


def load_yaml_config(file_path: str) -> dict[str, dict[str, str] | str]:
    """Reads the yaml file at the specified path and parses it into a dictionary.
    Args:
        file_path: The path of the yaml file to be loaded.
    Returns: The dictionary parsed from the specified YAML file.
    """
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        return yaml.safe_load(file)
