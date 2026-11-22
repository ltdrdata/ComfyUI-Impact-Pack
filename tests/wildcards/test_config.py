import logging
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("contents", [b"not an ini file", b"[unrelated]\nvalue=1"])
def test_unreadable_config_reports_fallback(tmp_path, monkeypatch, caplog, contents):
    path = tmp_path / "impact-pack.ini"
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


def test_missing_config_is_created_with_the_defaults(tmp_path, monkeypatch, caplog):
    path = tmp_path / "impact-pack.ini"
    monkeypatch.setattr(config, "config_path", str(path))
    monkeypatch.setattr(config, "cached_config", None)
    with caplog.at_level(logging.WARNING):
        created = config.get_config()
        assert path.read_text(encoding="utf-8").startswith("[default]\n")
        assert config.read_config() == created == config.default_config()
    assert not caplog.records


def test_config_that_cannot_be_created_falls_back_to_the_defaults(tmp_path, monkeypatch, caplog):
    path = tmp_path / "no such folder" / "impact-pack.ini"
    monkeypatch.setattr(config, "config_path", str(path))
    monkeypatch.setattr(config, "cached_config", None)
    with caplog.at_level(logging.WARNING):
        assert config.get_config() == config.default_config()
    assert not path.exists()
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "Failed to create impact-pack.ini" in caplog.text


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


def test_package_paths_have_no_parent_components(tmp_path, monkeypatch):
    """Windows does not resolve `..` inside `\\\\?\\` paths (#1266)."""
    resource_paths = [config.config_path, config.latent_letter_path, wildcards.wildcards_path]
    monkeypatch.setattr(config, "config_path", str(tmp_path / "missing.ini"))
    resource_paths.append(config.read_config()["custom_wildcards"])
    for path in resource_paths:
        assert ".." not in Path(path).parts, path
