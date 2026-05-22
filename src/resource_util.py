import os
from pathlib import Path

from runfiles import runfiles


def load_directory(dir_path: str | None = None) -> Path:
    """Resolve target directory seemlessly across multiple running environments.
    1. Standard CLI executable.
    2. 'bazel run //target' (Local Bazel workspace execution).
    3. Direct local 'pytest' calls.
    4. 'bazel test //target' (Isolated Bazel sandbox test suites).
    5. Standalone self-contained compiled Bazel binaries.

    Args:
        dir_path: The relatie path of the directory to be resolved. If None,
        defaults to the root of the project/working dir.
    Returns: The resolved directory path.

    """
    # anchor the working directory.
    base_working_dir: Path
    # Case: 'bazel run'
    if "BUILD_WORKDING_DIRECTORY" in os.environ:
        base_working_dir = Path(os.environ["BUILD_WORKING_DIRECTORY"])
    # case: 'bazel test'
    elif "TEST_SRCDIR" in os.environ:
        test_workspace = os.environ.get("TEST_WORKSPACE", "")
        base_working_dir = Path(os.environ["TEST_SRCDIR"]) / test_workspace
    # case: Standard python runtime or direct 'pytest'
    else:
        base_working_dir = Path.cwd()
    base_working_dir = base_working_dir.resolve()

    # if a relative directory was specified
    if dir_path is not None:
        raw_path = Path(dir_path)
        if raw_path.is_absolute():
            return raw_path.resolve()
        return (base_working_dir / raw_path).resolve()
    # if no directory path was specified default to project or working dir

    # case bazel env
    if "RUNFILES_DIR" in os.environ or "TEST_SRCDIR" in os.environ:
        rf = runfiles.Create()
        assert rf
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

    # case standard python runtime
    file_anchor = Path(__file__).resolve()
    if file_anchor.parent.name == "src":
        return file_anchor.parents[0]
    if file_anchor.parent.parent.name == "src":
        return file_anchor.parents[1]

    # global hard fallback
    return base_working_dir
