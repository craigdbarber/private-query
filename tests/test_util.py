"""A collection of utility functions for testing."""

import os
import shlex
import subprocess
import time
from pathlib import Path

import psutil


def load_test_resource(name: str) -> Path:
    """Load test resource matching the specified name. Handles cases of either
    running in bazel or via direct pytest invocation.

    Args:
        name: The name of the resource to be loaded. Assumes the name to be unique.

    Returns: The path of the requested resource.
    Raises: A FileNoneFoundError if the requested resource isn't found.

    """
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    project_path = Path(__file__).parent.parent

    # if running in bazel
    if runfiles_dir:
        runfiles_dir_path = Path(runfiles_dir)
        assert runfiles_dir_path is not None and runfiles_dir_path.exists()
        for resource in runfiles_dir_path.rglob(name):
            assert resource is not None and resource.exists()
            return resource
    # if running pytest directly
    elif project_path:
        resource = project_path / name
        assert resource is not None and resource.exists()
        return resource

    raise FileNotFoundError(f"Failed to load test resource: {name}.")


def start_local_ollama(host: str, home_dir: Path):
    """Start a local ollama server for testing.

    Args:
        host: The host URL for ollama configuration.
        home_dir: The home directory for ollama configuration.

    """
    if is_process_running("ollama"):
        return

    start_script_path = load_test_resource("scripts/start_ollama.sh")
    subprocess.run(  # noqa: S603
        [
            str(start_script_path),
            "--host",
            shlex.quote(host),
            "--ollama_home_dir",
            shlex.quote(str(home_dir)),
        ],
        check=True,
    )
    # verify ollama is running before returning
    while True:
        if is_process_running("ollama"):
            break
        time.sleep(2)


def is_process_running(name: str) -> bool:
    """Return whether a process with the specified name is running.

    Args:
        name: The name of the process.
    Returns: Whether the process is running.

    """
    return any(
        name in proc.info["name"].lower() for proc in psutil.process_iter(["name"])
    )


def stop_process(name: str) -> bool:
    """Attempt to stop the process with the specified name.

    Args:
        name: The name of the process to be stopped.
    Returns: Whether the process was successfully stopped.

    """
    stopped = False
    for proc in psutil.process_iter(["name", "pid"]):
        try:
            if name in proc.info["name"].lower():
                proc.kill()
                stopped = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # handles cases where process closes natually while loop is running or
            # requires admin rights
            continue

    return stopped
