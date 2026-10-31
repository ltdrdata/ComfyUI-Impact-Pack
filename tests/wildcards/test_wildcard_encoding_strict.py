"""Strict UTF-8 wildcard decoding checks against the fixture ground truth."""
import logging
import os
from pathlib import Path
import sys
import tempfile

import pytest
import yaml

# Use pytest's bootstrap before importing the wildcard module in script mode.
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

from impact import wildcards as wc  # noqa: E402
from impact.wildcards import (  # noqa: E402
    LazyWildcardLoader,
    load_txt_wildcard,
    load_yaml_wildcard,
    loaded_wildcards,
    read_wildcard_dict,
    wildcard_dict,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

KOREAN_FIXTURE = os.path.join(SAMPLES_DIR, "아름다운색.txt")
EMOJI_FIXTURE = os.path.join(SAMPLES_DIR, "test_encoding_emoji.txt")
SPECIAL_FIXTURE = os.path.join(SAMPLES_DIR, "test_encoding_special.txt")

KOREAN_EXPECTED = {
    "수놓은 별빛",
    "벚꽃 핑크",
    "강코랄",
    "옌로우",
    "챈메랄드",
    "챔무",
    "백설민주",
    "나부키하늘",
    "토미베이지",
    "율렌지",
    "블루지니",
    "캔디핑크",
}

EMOJI_EXPECTED = {
    "🌸🌺🌼🌻🌷🌹🥀🏵️💐🌾",
    "🌲🌳🌴🌵🌾🌿☘️🍀🍁🍂",
    "🌞🌝🌛🌜🌚🌕🌖🌗🌘🌑",
    "🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯",
    "🦁🐮🐷🐸🐵🙈🙉🙊🐒🐔",
    "🍎🍊🍋🍌🍉🍇🍓🍈🍒🍑",
    "🍕🍔🍟🌭🍿🧂🥓🥚🍳🧇",
    "❤️🧡💛💚💙💜🖤🤍🤎💔",
    "💕💞💓💗💖💘💝💟☮️✝️",
    "a 🌸 beautiful 🌺 garden 🌼 with 🌻 many 🌷 flowers 🌹",
}

YAML_KOREAN_EXPECTED = {"colors": ["빨강", "파랑", "초록"]}

# Re-used by the temp-file yaml tests; identical content to the on-disk fixture.
YAML_KOREAN_CONTENT = "colors:\n  - 빨강\n  - 파랑\n  - 초록\n"


# ---------- .txt load path coverage ----------

def test_korean_exact_set():
    """load_txt_wildcard: Korean 12-line fixture → exact set equality."""
    got = load_txt_wildcard(KOREAN_FIXTURE)
    assert set(got) == KOREAN_EXPECTED, (
        f"Korean wildcard mismatch.\n"
        f"  missing: {KOREAN_EXPECTED - set(got)!r}\n"
        f"  unexpected: {set(got) - KOREAN_EXPECTED!r}"
    )


def test_emoji_exact_set():
    """load_txt_wildcard: Emoji 10-line fixture → exact set equality."""
    got = load_txt_wildcard(EMOJI_FIXTURE)
    assert set(got) == EMOJI_EXPECTED, (
        f"Emoji wildcard mismatch.\n"
        f"  missing: {EMOJI_EXPECTED - set(got)!r}\n"
        f"  unexpected: {set(got) - EMOJI_EXPECTED!r}"
    )


def test_special_structural_len_canary():
    """load_txt_wildcard: special-chars fixture yields exactly 13 list entries.

    Under a mojibake decode, the 0x85 byte embedded in UTF-8 sequences for
    '∅' (U+2205 → E2 88 85) and 'υ' (U+03C5 → CF 85) is the Latin-1 line
    separator NEL, which splitlines() treats as a line break. A buggy decoder
    returns 15 entries; a strict UTF-8 decoder preserves the multibyte
    sequence intact and returns 13.
    """
    got = load_txt_wildcard(SPECIAL_FIXTURE)
    assert len(got) == 13, (
        f"Special wildcard length mismatch: expected 13 lines post-filter, "
        f"got {len(got)}. A count of 15 indicates Latin-1 phantom-split on "
        f"0x85 bytes inside U+2205/U+03C5 UTF-8 sequences."
    )


# ---------- .yaml load path coverage ----------

def test_yaml_korean_deterministic():
    """LazyWildcardLoader._load_yaml: temp Korean yaml → exact dict equality.

    Covers the `_load_yaml` branch of LazyWildcardLoader — one of the three
    yaml decode sites touched by the #1196 fix. Pre-fix, Korean content
    *happened* to load correctly because yaml.reader.ReaderError fires on
    the 0x80-0x9F control bytes produced by the ISO-8859-1 primary decode,
    triggering the UTF-8 fallback. The fix makes this deterministic — no
    reliance on PyYAML's Reader.check_printable quirk.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", delete=False
    ) as f:
        f.write(YAML_KOREAN_CONTENT)
        path = f.name
    try:
        loader = LazyWildcardLoader(path, "yaml")
        got = loader.get_data()
        assert got == YAML_KOREAN_EXPECTED, (
            f"LazyWildcardLoader._load_yaml Korean mismatch: got={got!r}"
        )
    finally:
        os.unlink(path)


def test_load_yaml_wildcard_korean():
    """load_yaml_wildcard: temp Korean yaml → loaded_wildcards['colors'] exact.

    Covers the module-level `load_yaml_wildcard()` function — another of
    the three yaml decode sites. The function populates the global
    `loaded_wildcards` dict; we verify the 'colors' key byte-equals the
    expected Korean list.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", delete=False
    ) as f:
        f.write(YAML_KOREAN_CONTENT)
        path = f.name
    try:
        loaded_wildcards.pop("colors", None)
        load_yaml_wildcard(path)
        got = loaded_wildcards.get("colors")
        assert got == YAML_KOREAN_EXPECTED["colors"], (
            f"load_yaml_wildcard Korean mismatch: got={got!r}"
        )
    finally:
        os.unlink(path)
        loaded_wildcards.pop("colors", None)


def test_read_wildcard_dict_yaml_branch():
    """read_wildcard_dict: tempdir containing Korean .yaml → wildcard_dict entry exact.

    Covers the inline yaml-decode branch of `read_wildcard_dict()` (the
    third yaml decode site). Writes a temp .yaml into a tempdir so the
    scan finds it without polluting the production wildcards directory.
    """
    with tempfile.TemporaryDirectory() as td:
        yaml_path = os.path.join(td, "test_yaml_korean.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(YAML_KOREAN_CONTENT)
        wildcard_dict.pop("colors", None)
        read_wildcard_dict(td)
        got = wildcard_dict.get("colors")
        assert got == YAML_KOREAN_EXPECTED["colors"], (
            f"read_wildcard_dict yaml-branch Korean mismatch: got={got!r}"
        )
        wildcard_dict.pop("colors", None)


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
            # The broken file must precede a good file, regardless of filesystem order.
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
