"""Provides functionality for seemlessly resolving runtime resources across multiple
running environments:
    1. Standard CLI executable.
    2. 'bazel run //target' (Local Bazel workspace execution).
    3. Direct local 'pytest' calls.
    4. 'bazel test //target' (Isolated Bazel sandbox test suites).
    5. Standalone self-contained compiled Bazel binaries.
"""

import os
from pathlib import Path

from loguru import logger
from runfiles import runfiles


def resolve_resource(path: str | Path) -> Path:
    """Resolve the specified resource.

    Args:
        path: The path of the resource to be resolved.
    Returns: The resolved resource.

    """
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path.resolve()
    resolved_path = (_get_base_working_dir() / raw_path).resolve()
    logger.debug(f"resolved_path: {resolved_path}")
    return resolved_path


def resolve_directory(dir_path: str | Path | None = None) -> Path:
    """Resolve the specified directory path if specified, otherwise resolves
    the base working dir.

    Args:
        dir_path: The relatie path of the directory to be resolved. If None,
        defaults to the root of the project/working dir.
    Returns: The resolved directory path.

    """
    # if a relative directory was specified
    if dir_path is not None:
        return resolve_resource(dir_path)

    # if no directory path was specified default to project or working dir
    # case bazel env
    if "RUNFILES_DIR" in os.environ or "TEST_SRCDIR" in os.environ:
        rf = runfiles.Create()
        if rf is None:
            raise RuntimeError("Bazel runtime files could not be created.")
        workspace_name = rf.CurrentRepository()
        if not workspace_name:
            workspace_name = os.environ.get("TEST_WORKSPACE", "")
        if not workspace_name:
            runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get(
                "TEST_SRCDIR"
            )
            if runfiles_dir:
                workspace_name = Path(runfiles_dir).name
        if workspace_name:
            workspace_root = rf.Rlocation(workspace_name)
            if workspace_root:
                return Path(workspace_root).resolve()
        # global bazel fallback if manifest hooks completely miss
        bazel_env_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
        if bazel_env_dir:
            return Path(bazel_env_dir).resolve()

    # global hard fallback
    return _get_base_working_dir()


def _get_base_working_dir() -> Path:
    # anchor the working directory.
    base_working_dir: Path
    # Case: 'bazel run'
    if "BUILD_WORKING_DIRECTORY" in os.environ:
        base_working_dir = Path(os.environ["BUILD_WORKING_DIRECTORY"])
    # case: 'bazel test'
    elif "TEST_SRCDIR" in os.environ:
        test_workspace = os.environ.get("TEST_WORKSPACE", "")
        base_working_dir = Path(os.environ["TEST_SRCDIR"]) / test_workspace
    # case: Standard python runtime or direct 'pytest'
    else:
        base_working_dir = Path.cwd()
    return base_working_dir.resolve()
