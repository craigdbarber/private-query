"""A collection of utility functions for testing."""

import shlex
import subprocess  # nosec
from pathlib import Path

import psutil
from loguru import logger

from resource_util import resolve_directory
from retry_util import random_exponential_retry


def start_local_ollama(host: str, home_dir: Path):
    """Start a local ollama server for testing.

    Args:
        host: The host URL for ollama configuration.
        home_dir: The home directory for ollama configuration.

    """
    if is_process_running("ollama"):
        return

    scripts_dir_path = resolve_directory("scripts")
    start_script_path = (scripts_dir_path / "start_ollama.sh").resolve()
    logger.info(
        f"Starting ollama service, host: {host} start_script_path: {start_script_path}"
    )

    try:
        subprocess.run(  # noqa: S603 # nosec
            [
                str(start_script_path),
                "--host",
                shlex.quote(host),
                "--ollama_home_dir",
                shlex.quote(str(home_dir)),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as cpe:
        logger.error(f"Failed to start ollama service: error: {cpe.stderr.decode()}")
        raise cpe

    # verify ollama is running before returning
    def wait_ollama_started():
        if not is_process_running("ollama"):
            raise TimeoutError("Timed out waiting for ollama service to start.")

    random_exponential_retry(wait_ollama_started)


def is_process_running(name: str) -> bool:
    """Return whether a process with the specified name is running.

    Args:
        name: The name of the process.
    Returns: Whether the process is running.

    """
    return any(
        name in proc.info["name"].lower() for proc in psutil.process_iter(["name"])
    )
