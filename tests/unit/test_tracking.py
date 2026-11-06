"""Reproducibility tags: git_dirty must reflect the code, not pipeline outputs."""

import subprocess

import pytest

from malariascope.tracking import source_is_dirty


def run(*args):
    subprocess.run(args, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("git", "init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "train.json").write_text("{}")
    (tmp_path / "dvc.lock").write_text("a")
    (tmp_path / "train.py").write_text("x = 1\n")
    run("git", "add", ".")
    run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return tmp_path


def test_clean_tree_is_not_dirty(repo):
    assert source_is_dirty() is False


def test_stage_output_deleted_by_dvc_is_not_dirty(repo):
    (repo / "results" / "train.json").unlink()
    (repo / "dvc.lock").write_text("b")
    assert source_is_dirty() is False


def test_changed_source_is_dirty(repo):
    (repo / "train.py").write_text("x = 2\n")
    assert source_is_dirty() is True
