"""Configuration regressions without importing the ComfyUI server."""
import logging
from pathlib import Path

import pytest

from impact import config


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
        "wildcard_cache_limit_mb": 50,
    }
    assert "Failed to read impact-pack.ini" in caplog.text
