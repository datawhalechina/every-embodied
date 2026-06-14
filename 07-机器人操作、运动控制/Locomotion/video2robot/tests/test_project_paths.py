from pathlib import Path

import pytest

import video2robot.utils as utils


def _use_tmp_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(utils, "DATA_DIR", data_dir)
    return data_dir


@pytest.mark.parametrize(
    "name",
    ["video_001", "中文项目", "demo project", "a.b-c_1"],
)
def test_resolve_project_dir_accepts_plain_project_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
):
    data_dir = _use_tmp_data_dir(monkeypatch, tmp_path)

    project_dir = utils.resolve_project_dir(name)

    assert project_dir == (data_dir / name).resolve()
    assert project_dir.parent == data_dir.resolve()


@pytest.mark.parametrize(
    "name",
    ["", "   ", ".", "..", "../outside", "foo/bar", r"foo\bar", "/tmp/outside", "a\x00b"],
)
def test_resolve_project_dir_rejects_path_like_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
):
    _use_tmp_data_dir(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        utils.resolve_project_dir(name)


def test_resolve_project_dir_rejects_existing_symlink_outside_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = _use_tmp_data_dir(monkeypatch, tmp_path)
    data_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (data_dir / "linked").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError):
        utils.resolve_project_dir("linked")


def test_resolve_project_dir_can_create_valid_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = _use_tmp_data_dir(monkeypatch, tmp_path)

    project_dir = utils.resolve_project_dir("demo", create=True)

    assert project_dir == (data_dir / "demo").resolve()
    assert project_dir.is_dir()


@pytest.mark.parametrize("filename", ["original.mp4", "robot_motion.pkl", "smplx_tracks.json"])
def test_resolve_project_file_accepts_single_file_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
):
    data_dir = _use_tmp_data_dir(monkeypatch, tmp_path)

    file_path = utils.resolve_project_file("demo", filename)

    assert file_path == (data_dir / "demo" / filename).resolve()
    assert file_path.parent == (data_dir / "demo").resolve()


@pytest.mark.parametrize(
    "filename",
    ["", "   ", ".", "..", "../secret.txt", "nested/file.txt", r"nested\file.txt", "/etc/passwd"],
)
def test_resolve_project_file_rejects_path_like_filenames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
):
    _use_tmp_data_dir(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        utils.resolve_project_file("demo", filename)


def test_ensure_project_dir_validates_name_but_preserves_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = _use_tmp_data_dir(monkeypatch, tmp_path)

    named_project = utils.ensure_project_dir(name="demo")
    explicit_project = utils.ensure_project_dir(project_path=tmp_path / "external-project")

    assert named_project == (data_dir / "demo").resolve()
    assert named_project.is_dir()
    assert explicit_project == tmp_path / "external-project"
    assert explicit_project.is_dir()

    with pytest.raises(ValueError):
        utils.ensure_project_dir(name="../outside")
