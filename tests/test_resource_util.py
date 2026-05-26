"""A test suite for resource_util."""

import sys
from pathlib import Path

import pytest

from resource_util import resolve_directory, resolve_resource


def test_resolve_resource_resolves_file(tmp_path: Path):
    """Tests successfully resolves a file."""
    tmp_file = (tmp_path / "test_file").resolve()
    tmp_file.touch()
    resolved_tmp = resolve_resource(tmp_file)
    assert resolved_tmp is not None
    assert resolved_tmp.exists()
    assert tmp_file == resolved_tmp


def test_resolve_directory_resolves_dir(tmp_path_factory: pytest.TempPathFactory):
    """Tests successfully resolves a directory."""
    tmp_dir = tmp_path_factory.mktemp("tmp_dir")
    resolved_dir = resolve_directory(tmp_dir)
    assert resolved_dir is not None
    assert resolved_dir.exists()
    assert resolved_dir.is_dir()
    assert tmp_dir.resolve() == resolved_dir


def test_resolve_directory_resolves_working_dir():
    """Tests successfully resolves working dir."""
    working_dir = resolve_directory()
    assert working_dir is not None
    assert working_dir.exists()
    assert working_dir.is_dir()


def test_resolve_directory_bazel_env(_mock_bazel_repo_working_dir: Path):
    """Test successfully resolves working dir for bazel env."""
    working_dir = resolve_directory()
    assert working_dir is not None
    assert working_dir == _mock_bazel_repo_working_dir


@pytest.fixture
def _mock_bazel_repo_working_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    # create a dummy repo mapping file that bazel utilities look for
    repo_mapping_file = (tmp_path / "repo_mapping").resolve()
    # Content format: <caller_repo>,<target_repo_canonical_name>,<apparent_name>
    repo_mapping_file.write_text(", ,\n")

    # create a mock repo root folder
    runfiles_dir = (tmp_path / "runfiles").resolve()
    repo_working_dir = runfiles_dir
    repo_working_dir.mkdir(parents=True, exist_ok=True)

    # tell runfiles lib where to find manifest and runfiles dir
    manifest_file = tmp_path / "MANIFEST_FILE"
    manifest_file.touch()
    monkeypatch.setenv("RUNFILES_MANIFEST_FILE", str(manifest_file))
    monkeypatch.setenv("RUNFILES_DIR", str(runfiles_dir))

    # inject bzlmod repo mapping variables
    monkeypatch.setenv("BAZEL_REPO_MAPPING_CHUNKS", "1")
    monkeypatch.setenv("REPOSITORY_MAPPING_FILE", str(repo_mapping_file))

    # mock the execution call stack so bazel tracks the frame of origin,
    # forcing CurrentRepository() to align with our fake caller location
    monkeypatch.setattr(sys, "argv", [f"{repo_working_dir}/bin/main_cli.py"])
    return repo_working_dir
