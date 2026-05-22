"""A collection of utility functions for testing."""

import os
import shlex
import subprocess
import time
from pathlib import Path

import psutil
from runfiles import runfiles


def load_test_data_dir(data_dir: str | None = None) -> Path:
    """Load a target directory supporting execution from both bazel and pytest.

    Args:
        data_dir: The relative path of the directory to be loaded (e.g., "tests/data").
                  If None or empty, returns the project root directory.
    Returns: The requested data dir, or the project root directory.

    Raises:
        FileNotFoundError: If the specified relative path does not exist.

    """
    # check if running under bazel
    if "RUNFILES_DIR" in os.environ or "RUNFILES_MANIFEST_FILE" in os.environ:
        rf = runfiles.Create()
        assert rf
        current_repo = rf.CurrentRepository()
        repo_root = current_repo if current_repo else "_main"
        if data_dir:
            runfiles_path = rf.Rlocation(f"{repo_root}/{data_dir}")
            if (
                runfiles_path
                and os.path.exists(runfiles_path)
                and os.path.isdir(runfiles_path)
            ):
                return Path(runfiles_path).resolve()
        else:
            root_path = rf.Rlocation(repo_root)
            if root_path and os.path.exists(root_path):
                return Path(root_path).resolve()

    # fallback to standard pytest exeuction from the project root
    project_root = Path(__file__).parent.parent
    if data_dir:
        data_dir_path = project_root / data_dir
        if data_dir_path.exists() and data_dir_path.is_dir():
            return data_dir_path.resolve()
    else:
        return project_root.resolve()

    # if the specified data directory could not be found, raise an error
    raise FileNotFoundError(f"Failed to load test directory: {data_dir}")


def start_local_ollama(host: str, home_dir: Path):
    """Start a local ollama server for testing.

    Args:
        host: The host URL for ollama configuration.
        home_dir: The home directory for ollama configuration.

    """
    if is_process_running("ollama"):
        return

    scripts_dir_path = load_test_data_dir("scripts")
    start_script_path = (scripts_dir_path / "start_ollama.sh").resolve()
    # start_script_path = load_test_resource("scripts/start_ollama.sh")
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
