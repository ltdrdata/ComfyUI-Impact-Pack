"""Configuration regressions without importing the ComfyUI server."""
import logging
import ntpath
from pathlib import Path
import sys
import types

import pytest

import impact
from impact import config, wildcards


@pytest.mark.parametrize("model", [None, "sam_vit_h_4b8939.pth"])
def test_optional_model_preserves_user_settings(tmp_path, monkeypatch, caplog, model):
    custom = tmp_path / "custom wildcards"
    custom.mkdir()
    path = tmp_path / "impact-pack.ini"
    path.write_text(
        '[default]\nsam_editor_cpu = True\ndisable_gpu_opencv = False\n'
        f'custom_wildcards = "{custom}"\nwildcard_cache_limit_mb = 0.25\n'
        + (f'sam_editor_model = {model}\n' if model else ''),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "config_path", str(path))
    with caplog.at_level(logging.WARNING):
        result = config.read_config()
    assert result == {
        "sam_editor_cpu": True,
        "sam_editor_model": model or "sam_vit_b_01ec64.pth",
        "custom_wildcards": str(custom),
        "disable_gpu_opencv": False,
        "gpu_gaussian_blur": False,
        "wildcard_cache_limit_mb": 0.25,
    }
    assert not caplog.records


@pytest.mark.parametrize("contents", [None, b"not an ini file", b"[unrelated]\nvalue=1"])
def test_unreadable_config_reports_fallback(tmp_path, monkeypatch, caplog, contents):
    path = tmp_path / "impact-pack.ini"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr(config, "config_path", str(path))
    with caplog.at_level(logging.ERROR):
        result = config.read_config()
    assert result == {
        "sam_editor_cpu": False,
        "sam_editor_model": "sam_vit_b_01ec64.pth",
        "custom_wildcards": str((Path(config.my_path) / "../../custom_wildcards").resolve()),
        "disable_gpu_opencv": True,
        "gpu_gaussian_blur": False,
        "wildcard_cache_limit_mb": 50,
    }
    assert "Failed to read impact-pack.ini" in caplog.text


@pytest.mark.parametrize("value, enabled", [(None, False), ("False", False), ("tRuE", True)])
def test_gpu_gaussian_blur_config_roundtrip(tmp_path, monkeypatch, value, enabled):
    path = tmp_path / "impact-pack.ini"
    path.write_text(
        f"[default]\ncustom_wildcards = {tmp_path}\nsam_editor_cpu = True\n"
        + (f"gpu_gaussian_blur = {value}\n" if value is not None else ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "config_path", str(path))
    monkeypatch.setattr(config, "cached_config", None)
    before = config.get_config().copy()
    assert before["gpu_gaussian_blur"] is enabled
    assert before["sam_editor_cpu"] is True
    config.write_config()
    assert config.read_config() == before


@pytest.fixture(params=[
    pytest.param((r"C:\ComfyUI\custom_nodes\Impact Pack", False), id="drive"),
    pytest.param((r"\\server\share\ComfyUI\custom_nodes\Impact Pack", False), id="unc"),
    pytest.param((r"\\?\C:\ComfyUI\custom_nodes\Impact Pack", False), id="extended-drive"),
    pytest.param((r"\\?\UNC\server\share\ComfyUI\custom_nodes\Impact Pack", False), id="extended-unc"),
    pytest.param(("\\\\?\\C:\\" + "long directory\\" * 20 + "Impact Pack", False), id="long-extended-drive"),
    pytest.param(("\\\\?\\UNC\\server\\share\\" + "long directory\\" * 20 + "Impact Pack", False), id="long-extended-unc"),
    pytest.param(("\\\\?\\C:\\" + "long directory\\" * 20 + "Impact Pack", True), id="extended-drive-python310"),
    pytest.param(("\\\\?\\UNC\\server\\share\\" + "long directory\\" * 20 + "Impact Pack", True), id="extended-unc-python310"),
])
def windows_package(request, tmp_path, monkeypatch):
    """Load the real modules with Windows paths and file I/O backed by tmp_path."""
    root, preserve_extended_parents = request.param
    paths = types.SimpleNamespace(**vars(ntpath))
    # Python 3.10's ntpath preserves literal extended paths; newer versions
    # normalize them. Neither version normalizes paths passed directly to open.
    if preserve_extended_parents:
        paths.abspath = lambda path: path if path.startswith("\\\\?\\") else ntpath.abspath(path)

    def local_path(path):
        if path.startswith("\\\\?\\") and ".." in path.split("\\"):
            raise FileNotFoundError(path)
        path = ntpath.normpath(path)
        if not path.startswith(root + "\\"):
            raise FileNotFoundError(path)
        return tmp_path.joinpath(*path[len(root) + 1:].split("\\"))

    def open_file(path, *args, **kwargs):
        return local_path(path).open(*args, **kwargs)

    def exists(path):
        try:
            return local_path(path).exists()
        except FileNotFoundError:
            return False

    paths.exists = exists
    loaded = []
    for original in (config, wildcards):
        source_path = Path(original.__file__)
        source = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
        module = types.ModuleType(original.__name__)
        module.__file__ = ntpath.join(root, "modules", "impact", source_path.name)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "os", types.SimpleNamespace(path=paths))
            if loaded:
                patch.setattr(impact, "config", loaded[0])
            exec(source, module.__dict__)
        module.open = open_file
        loaded.append(module)
    monkeypatch.setattr(config.configparser, "open", open_file, raising=False)
    return root, loaded[0], loaded[1]


def test_windows_latent_and_legacy_config_access(windows_package, tmp_path):
    _, cfg, _ = windows_package
    for path, relative_path, contents in (
        (cfg.latent_letter_path, "latent.png", b"latent image"),
        (cfg.old_config_path, "modules/impact/impact-pack.ini", b"[default]\n"),
    ):
        resource = tmp_path / relative_path
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_bytes(contents)
        with cfg.open(path, "rb") as file:
            assert file.read() == contents


def test_windows_config_read_write(windows_package, tmp_path, caplog):
    root, cfg, _ = windows_package
    (tmp_path / "user wildcards").mkdir()
    custom = ntpath.join(root, "user wildcards")
    (tmp_path / "impact-pack.ini").write_text(
        '[default]\nsam_editor_cpu = True\nsam_editor_model = sam_vit_h_4b8939.pth\n'
        f'custom_wildcards = "{custom}"\ndisable_gpu_opencv = False\n'
        'gpu_gaussian_blur = True\nwildcard_cache_limit_mb = 0.25\n',
        encoding="utf-8",
    )
    expected = {
        "sam_editor_cpu": True,
        "sam_editor_model": "sam_vit_h_4b8939.pth",
        "custom_wildcards": custom,
        "disable_gpu_opencv": False,
        "gpu_gaussian_blur": True,
        "wildcard_cache_limit_mb": 0.25,
    }
    assert cfg.get_config() == expected
    cfg.write_config()
    assert cfg.read_config() == expected
    assert not caplog.records


@pytest.mark.parametrize("contents", [None, "[default]\n", "[default]\ncustom_wildcards = missing\n"])
def test_windows_default_wildcard_paths(windows_package, tmp_path, contents):
    _, cfg, wc = windows_package
    if contents is not None:
        (tmp_path / "impact-pack.ini").write_text(contents, encoding="utf-8")
    for directory, text in (("wildcards", "red\nblue\n"), ("custom_wildcards", "green\nyellow\n")):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "sample.txt").write_text(text, encoding="utf-8")

    custom = cfg.read_config()["custom_wildcards"]
    assert wc.load_txt_wildcard(ntpath.join(wc.wildcards_path, "sample.txt")) == ["red", "blue"]
    assert wc.load_txt_wildcard(ntpath.join(custom, "sample.txt")) == ["green", "yellow"]
