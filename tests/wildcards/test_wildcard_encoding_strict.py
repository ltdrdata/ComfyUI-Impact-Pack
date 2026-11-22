import logging
from pathlib import Path
import sys

import pytest
import yaml

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

from impact import wildcards as wc  # noqa: E402


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "ISO-8859-1"])
@pytest.mark.parametrize("file_type", ["txt", "yaml"])
@pytest.mark.parametrize("loader", ["direct", "lazy", "eager", "on_demand"])
def test_decoding_across_loaders(tmp_path, monkeypatch, caplog, encoding, file_type, loader):
    expected = ["café", "résumé"]
    content = "# header\n\ncafé\nrésumé\n" if file_type == "txt" else "encoded:\n  - café\n  - résumé\n"
    path = tmp_path / f"encoded.{file_type}"
    path.write_bytes(content.encode(encoding))
    monkeypatch.setattr(wc, "wildcard_dict", {})
    monkeypatch.setattr(wc, "loaded_wildcards", {})

    with caplog.at_level(logging.WARNING):
        if loader == "direct":
            result = wc.load_txt_wildcard(path) if file_type == "txt" else wc.load_yaml_wildcard(path)
        elif loader == "lazy":
            result = wc.LazyWildcardLoader(path, file_type).get_data()
            if file_type == "yaml":
                result = result["encoded"]
        else:
            wc.read_wildcard_dict(tmp_path, on_demand=loader == "on_demand")
            result = list(wc.wildcard_dict["encoded"])

    assert result == expected
    if encoding == "ISO-8859-1":
        assert len(caplog.records) == 1
        assert "ISO-8859-1 fallback" in caplog.text
        assert str(path) in caplog.text
    else:
        assert not caplog.records


@pytest.fixture
def wildcard_loading(monkeypatch):
    for name in ("wildcard_dict", "loaded_wildcards", "available_wildcards"):
        monkeypatch.setattr(wc, name, {})
    monkeypatch.setattr(wc, "_on_demand_mode", False)
    walk = wc.os.walk

    def ordered_walk(*args, **kwargs):
        for root, directories, files in walk(*args, **kwargs):
            yield root, directories, sorted(files)

    monkeypatch.setattr(wc.os, "walk", ordered_walk)


@pytest.mark.parametrize("on_demand", [False, True])
def test_bad_file_is_skipped_and_named(tmp_path, wildcard_loading, caplog, on_demand):
    (tmp_path / "a_good.txt").write_text("red\nblue\n", encoding="utf-8")
    bad = tmp_path / "b_broken.yaml"
    bad.write_text("colors: [unclosed\n", encoding="utf-8")
    (tmp_path / "c_empty.yaml").write_text("", encoding="utf-8")
    (tmp_path / "d_good.yaml").write_text("shapes:\n  - circle\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        wc.read_wildcard_dict(tmp_path, on_demand=on_demand)

    assert list(wc.wildcard_dict["a_good"]) == ["red", "blue"]
    assert list(wc.wildcard_dict["shapes"]) == ["circle"]
    assert len(caplog.records) == 1
    assert str(bad) in caplog.text
    assert "ParserError" in caplog.text


@pytest.mark.parametrize("on_demand", [False, True], ids=["eager", "on_demand"])
@pytest.mark.parametrize("bad_directory", ["builtin", "custom"])
@pytest.mark.parametrize("contents", [b"colors: [unclosed\n", b"\x00\x05\x16\x07MacOS\x00", None],
                         ids=["syntax", "binary", "read_error"])
def test_wildcard_load_continues_after_bad_file(tmp_path, wildcard_loading, monkeypatch, caplog,
                                              on_demand, bad_directory, contents):
    builtin, custom = tmp_path / "builtin", tmp_path / "custom"
    for label, directory in (("builtin", builtin), ("custom", custom)):
        directory.mkdir()
        (directory / f"a_{label}.txt").write_text(f"{label}-before\n", encoding="utf-8")
        (directory / "c_empty.yaml").write_text("", encoding="utf-8")
        (directory / "d_after.yml").write_text(f"{label}_after: [{label}-after]\n", encoding="utf-8")
    bad = tmp_path / bad_directory / "b_bad.yaml"
    bad.write_bytes(contents if contents is not None else b"unreadable: [value]\n")
    if contents is None:
        read_text = wc._read_wildcard_text

        def unreadable(path):
            if Path(path) == bad:
                raise OSError("wildcard fixture is unreadable")
            return read_text(path)

        monkeypatch.setattr(wc, "_read_wildcard_text", unreadable)
    monkeypatch.setattr(wc, "wildcards_path", str(builtin))
    monkeypatch.setattr(wc.config, "get_config", lambda: {
        "custom_wildcards": str(custom), "wildcard_cache_limit_mb": 0 if on_demand else 50})

    with caplog.at_level(logging.WARNING):
        wc.wildcard_load()

    assert wc.is_on_demand_mode() is on_demand
    assert wc.process("__a_builtin__ __builtin_after__ __a_custom__ __custom_after__", seed=7) == (
        "builtin-before builtin-after custom-before custom-after")
    assert len(caplog.records) == 1
    assert str(bad) in caplog.text


@pytest.mark.parametrize("lazy", [False, True], ids=["direct", "lazy"])
@pytest.mark.parametrize("contents", [b"colors: [unclosed\n", b"\x00\x05\x16\x07MacOS\x00"], ids=["syntax", "binary"])
def test_single_file_loader_still_raises(tmp_path, lazy, contents):
    bad = tmp_path / "bad.yaml"
    bad.write_bytes(contents)
    with pytest.raises(yaml.YAMLError):
        if lazy:
            wc.LazyWildcardLoader(bad, "yaml").get_data()
        else:
            wc.load_yaml_wildcard(bad)


def test_invalid_yaml_is_not_redecoded_as_latin1(tmp_path, caplog):
    path = tmp_path / "invalid.yaml"
    path.write_text("colors: [\ufffe]\n", encoding="utf-8")
    with pytest.raises(yaml.reader.ReaderError):
        wc.load_yaml_wildcard(path)
    assert "ISO-8859-1 fallback" not in caplog.text
