"""Behavior and existing-workflow regressions for LoraWildcardSource."""
import asyncio
import importlib.util
import json
import ntpath
from pathlib import Path
import random
import shutil
import sys
import types

from aiohttp import web
import pytest

from impact import config, wildcards


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cache_limit", [0, 50], ids=["on_demand", "full_cache"])
def test_shipped_examples_preserve_existing_wildcard_results(tmp_path, monkeypatch, cache_limit):
    bundled = tmp_path / "wildcards"
    bundled.mkdir()
    (bundled / "styles.txt").write_text("watercolor\n", encoding="utf-8")
    custom = tmp_path / "custom_wildcards"
    custom.mkdir()
    monkeypatch.setattr(config, "cached_config", {
        "custom_wildcards": str(custom), "wildcard_cache_limit_mb": cache_limit,
    })
    monkeypatch.setattr(wildcards, "wildcards_path", str(bundled))
    for name in ("wildcard_dict", "available_wildcards", "loaded_wildcards"):
        monkeypatch.setattr(wildcards, name, {})
    monkeypatch.setattr(wildcards, "_on_demand_mode", False)
    wildcards.wildcard_load()
    assert wildcards._on_demand_mode == (cache_limit == 0)
    before = wildcards.process("portrait, __styles__", seed=42)
    shutil.copytree(ROOT / "custom_wildcards", custom, dirs_exist_ok=True)
    wildcards.wildcard_load()
    assert wildcards.process("portrait, __styles__", seed=42) == before == "portrait, watercolor"


@pytest.fixture
def source(tmp_path, monkeypatch):
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=web.RouteTableDef()))
    monkeypatch.setitem(sys.modules, "server", server)
    spec = importlib.util.spec_from_file_location("review_wildcard_source", ROOT / "modules/impact/wildcard_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = tmp_path / "pack"
    custom = pack / "custom_wildcards"
    custom.mkdir(parents=True)
    # Keep the package-default directory and configured directory equal unless
    # the test specifically exercises a relocated wildcard collection.
    monkeypatch.setattr(module, "__file__", str(pack / "modules/impact/wildcard_source.py"))
    monkeypatch.setattr(config, "cached_config", {"custom_wildcards": str(custom)})
    monkeypatch.setattr(wildcards, "wildcard_dict", {"color": ["blue"]})
    monkeypatch.setattr(wildcards, "_on_demand_mode", False)
    return module, custom


@pytest.fixture
def model_calls(monkeypatch):
    import folder_paths
    import nodes

    calls = []

    class Loader:
        def load_lora(self, model, clip, name, model_weight, clip_weight):
            calls.append((name, model_weight, clip_weight))
            return model + (name,), clip + (name,)

    class Encoder:
        def encode(self, clip, text):
            return ([text],)

    class Concat:
        def concat(self, first, second):
            return (first + second,)

    monkeypatch.setattr(nodes, "LoraLoader", Loader, raising=False)
    monkeypatch.setattr(nodes, "CLIPTextEncode", Encoder, raising=False)
    monkeypatch.setattr(nodes, "ConditioningConcat", Concat, raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda kind: [f"{name}.safetensors" for name in ("A", "B", "Base")], raising=False)
    return calls


def items(module, category):
    response = asyncio.run(module.get_items_handler(types.SimpleNamespace(query={"category": category})))
    return json.loads(response.text)


def build(module, **kwargs):
    values = dict(model=(), clip=(), category="styles", lora_wildcard="A", mode="fixed", strength=0.8, base_prompt="base", seed=0)
    return module.LoraWildcardSource().build(**(values | kwargs))


@pytest.mark.parametrize("tag", ["<lora:A:0.7:0.4>", "<lora:A:-0.7>", "<lora:A>", "<lora:A:1e-2>"])
def test_existing_lora_syntax_is_selectable(source, tag):
    module, custom = source
    (custom / "styles.txt").write_text("portrait, " + tag)
    assert items(module, "styles") == ["A"]


def test_configured_directory_drives_schema_route_and_execution(source, tmp_path, monkeypatch, model_calls):
    module, custom = source
    (custom / "local.txt").write_text("local, <lora:B:1>")
    external = tmp_path / "external"
    external.mkdir()
    (external / "remote.txt").write_text("external, <lora:A:1>")
    monkeypatch.setattr(config, "cached_config", {"custom_wildcards": str(external)})
    assert module.LoraWildcardSource.INPUT_TYPES()["required"]["category"][0] == ["remote"]
    assert items(module, "remote") == ["A"]
    assert "external" in build(module, category="remote")[3][0]


def test_full_entry_and_extra_loras_survive_encoding(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text("portrait, <lora:A:0.7>, __color__, <lora:B:-0.3:0.4> BREAK landscape")
    result = build(module, base_prompt="base, <lora:Base:0.2:0.3>")
    assert model_calls == [("Base.safetensors", 0.2, 0.3), ("A.safetensors", 0.8, 0.8), ("B.safetensors", -0.3, 0.4)]
    assert result[3] == ["base, portrait, __color__, BREAK landscape"]
    assert result[2] == [["base, portrait, __color__,", "landscape"]]


def test_simple_entry_keeps_original_clean_text_and_conditioning(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text("portrait, <lora:A:0.7>\n")
    result = build(module, base_prompt="base, ")
    assert result[3] == ["base, portrait"]
    assert result[2] == [["base, portrait"]]


def test_empty_category_still_processes_base_prompt_loras(source, model_calls):
    module, _ = source
    result = build(module, base_prompt="__color__, <lora:Base:0.2:0.3>")
    assert model_calls == [("Base.safetensors", 0.2, 0.3)]
    assert result[3] == ["__color__"]


@pytest.mark.parametrize("contents,prompt,sections", [
    (None, "a flower, ", ["a flower,"]),
    ("", " first,,  BREAK\nsecond, \t ", ["first,,", "second,"]),
    ("plain text\n", " \n\t ", [""]),
    (None, "", [""]),
])
def test_empty_category_preserves_plain_prompt(source, model_calls, contents, prompt, sections):
    module, custom = source
    if contents is not None:
        (custom / "styles.txt").write_text(contents)
    result = build(module, base_prompt=prompt)
    assert result[3] == [prompt]
    assert result[2] == [sections]
    assert result[0] == result[1] == [()]
    assert result[4:] == (["no_lora"], ["no_styles"])
    assert model_calls == []


@pytest.mark.parametrize("new_tag", ["<lora:A:-0.5>", "<lora:A>", "<lora:A:0.7:0.2>", "<lora:A:1e-2>"])
def test_extended_syntax_preserves_saved_label_targets(source, model_calls, new_tag):
    module, custom = source
    (custom / "styles.txt").write_text(
        f"new trigger, {new_tag}\nold first, <lora:A:0.7>\nold second, <lora:A:0.8>\n")
    assert items(module, "styles") == ["A", "A (1)", "A (2)"]
    for label, text in [("A", "base, old first"), ("A (1)", "base, old second"), ("A (2)", "base, new trigger")]:
        result = build(module, lora_wildcard=label)
        assert result[3] == [text]
        assert result[2] == [[text]]
    assert model_calls == [("A.safetensors", 0.8, 0.8)] * 3


def test_existing_primary_tag_keeps_its_label_and_strength(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text("<lora:Base:-0.3:0.4>, portrait, <lora:A:0.7>\n")
    assert items(module, "styles") == ["A"]
    result = build(module)
    assert model_calls == [("Base.safetensors", -0.3, 0.4), ("A.safetensors", 0.8, 0.8)]
    assert result[3] == ["base, portrait"]
    assert result[4] == ["A"]


def test_new_names_cannot_reuse_saved_duplicate_labels(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text(
        "old first, <lora:A:0.7>\nold second, <lora:A:0.8>\nnew trigger, <lora:A (1):-0.5>\n")
    assert items(module, "styles") == ["A", "A (1)", "A (1) (1)"]
    assert build(module, lora_wildcard="A (1)")[3] == ["base, old second"]


@pytest.mark.parametrize("mode,seed,expected", [
    ("random", 5, ["B"]), ("sequential", 2, ["A"]), ("batch", 0, ["A", "B"]),
])
def test_automatic_modes_keep_original_candidates(source, model_calls, mode, seed, expected):
    module, custom = source
    (custom / "styles.txt").write_text(
        "new negative, <lora:A:-0.5>\nold first, <lora:A:0.7>\n"
        "new dual, <lora:B:0.7:0.2>\nold second, <lora:B:0.8>\n"
        "new weightless, <lora:A>\nnew exponent, <lora:B:1e-2>\n")
    result = build(module, mode=mode, seed=seed, lora_wildcard="A (1)")
    texts = {"A": "base, old first", "B": "base, old second"}
    assert result[3] == [texts[name] for name in expected]
    assert result[2] == [[texts[name]] for name in expected]
    assert result[5] == expected
    assert [len(output) for output in result] == [len(expected)] * 6
    assert model_calls == [(name + ".safetensors", 0.8, 0.8) for name in expected]


@pytest.mark.parametrize("mode,selection", [
    ("random", "A"), ("sequential", "A"), ("batch", "A"),
    ("fixed", "(styles.txt not found)"), ("fixed", ""),
])
def test_modes_preserve_empty_legacy_selection(source, model_calls, mode, selection):
    module, custom = source
    (custom / "styles.txt").write_text("new negative, <lora:A:-0.5>\nnew weightless, <lora:B>\n")
    result = build(module, mode=mode, lora_wildcard=selection, base_prompt="base, ")
    assert result[3] == ["base, "]
    assert result[2] == [["base,"]]
    assert result[4:] == (["no_lora"], ["no_styles"])
    assert model_calls == []


@pytest.mark.parametrize("trigger,expected", [
    (",, portrait", "base,, portrait"), (", , portrait", "base, , portrait"),
    (",portrait", "base,portrait"), (",,, portrait", "base,, portrait"),
])
def test_entry_keeps_original_leading_punctuation(source, model_calls, trigger, expected):
    module, custom = source
    (custom / "styles.txt").write_text(trigger + " <lora:A:0.7>\n")
    result = build(module)
    assert result[3] == [expected]
    assert result[2] == [[expected]]
    assert model_calls == [("A.safetensors", 0.8, 0.8)]


def test_windows_cross_drive_link_preserves_valid_categories(source, monkeypatch, model_calls):
    module, _ = source
    root = r"C:\wildcards"
    valid = ntpath.join(root, "good.txt")
    outside = r"D:\external\styles.txt"

    def realpath(path):
        return outside if path == ntpath.join(root, "external.txt") else path

    def read(path):
        assert path == valid, "The excluded file must never be read"
        return ["portrait, <lora:A:0.7>"]

    # Resolve a Windows link without requiring a Windows filesystem. The
    # drive comparison itself uses the real Windows standard-library code.
    monkeypatch.setattr(module, "os", types.SimpleNamespace(
        path=types.SimpleNamespace(realpath=realpath, join=ntpath.join, splitext=ntpath.splitext,
                                   commonpath=ntpath.commonpath, isfile=lambda path: path in (valid, outside)),
        listdir=lambda path: ["external.txt", "good.txt"],
    ))
    monkeypatch.setattr(config, "cached_config", {"custom_wildcards": root})
    monkeypatch.setattr(wildcards, "load_txt_wildcard", read)
    schema = module.LoraWildcardSource.INPUT_TYPES()["required"]
    assert schema["category"][0] == ["good"]
    assert schema["lora_wildcard"][0] == ["A"]
    assert items(module, "external") == []
    assert build(module, category="external")[3] == ["base"]
    assert model_calls == []
    assert items(module, "good") == ["A"]
    assert build(module, category="good")[3] == ["base, portrait"]
    assert model_calls == [("A.safetensors", 0.8, 0.8)]


def test_pose_category_uses_the_same_file_in_ui_and_execution(source, model_calls):
    module, custom = source
    (custom / "pose.txt").write_text("singular, <lora:A:1>")
    (custom / "poses.txt").write_text("plural, <lora:B:1>")
    assert items(module, "pose") == ["A"]
    assert "singular" in build(module, category="pose")[3][0]


@pytest.mark.parametrize("category,target", [
    ("pose", "poses"), ("poses", "pose"), ("style", "styles"),
    ("styles", "estilos_arte"), ("estilo", "styles"), ("estilos_arte", "styles"),
    ("style", "estilos_arte"), ("estilo", "estilos_arte"),
])
def test_saved_category_aliases_keep_their_fallback(source, model_calls, category, target):
    module, custom = source
    (custom / (target + ".txt")).write_text("legacy target, <lora:A:0.7>\n")
    assert items(module, category) == ["A"]
    result = build(module, category=category)
    assert result[3] == ["base, legacy target"]
    assert result[2] == [["base, legacy target"]]
    assert model_calls == [("A.safetensors", 0.8, 0.8)]


def test_category_alias_prefers_exact_file_then_original_fallback_order(source, model_calls):
    module, custom = source
    for category, name in [("estilo", "A"), ("styles", "B"), ("estilos_arte", "Base")]:
        (custom / (category + ".txt")).write_text(f"{category}, <lora:{name}:0.7>\n")
    assert items(module, "estilo") == ["A"]
    assert build(module, category="estilo")[3] == ["base, estilo"]
    (custom / "estilo.txt").unlink()
    assert items(module, "estilo") == ["B"]
    assert build(module, category="estilo", lora_wildcard="B")[3] == ["base, styles"]
    assert model_calls == [("A.safetensors", 0.8, 0.8), ("B.safetensors", 0.8, 0.8)]


def test_alias_fallback_cannot_escape_configured_directory(source, tmp_path, model_calls):
    module, custom = source
    outside = tmp_path / "outside.txt"
    outside.write_text("private, <lora:A:0.7>\n")
    (custom / "styles.txt").symlink_to(outside)
    assert items(module, "style") == []
    assert build(module, category="style")[3] == ["base"]
    assert model_calls == []


@pytest.mark.parametrize("kind", ["relative", "absolute", "symlink"])
def test_category_cannot_read_outside_configured_directory(source, tmp_path, kind, model_calls):
    module, custom = source
    outside = tmp_path / "outside.txt"
    outside.write_text("private, <lora:A:1>")
    if kind == "symlink":
        (custom / "outside.txt").symlink_to(outside)
        category = "outside"
    elif kind == "absolute":
        category = str(outside.with_suffix(""))
    else:
        category = "../../outside"
    assert items(module, category) == []
    result = build(module, category=category)
    assert "private" not in result[3][0]
    assert model_calls == []


def test_category_decoding_matches_existing_wildcard_loader(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_bytes(b"caf\xe9, <lora:A:1>\n")
    assert items(module, "styles") == ["A"]
    result = build(module)
    assert result[3] == ["base, café"]
    assert result[2] == [["base, café"]]


@pytest.mark.parametrize("mode,seed,expected", [
    ("fixed", 0, ["B"]), ("sequential", 1, ["B"]), ("sequential", 2, ["A"]),
    ("random", 0, ["B"]), ("random", 1, ["A"]), ("batch", 1, ["A", "B"]),
])
def test_modes_are_deterministic_and_keep_models_separate(source, model_calls, mode, seed, expected):
    module, custom = source
    (custom / "styles.txt").write_text("first, <lora:A:1>\nsecond, <lora:B:1>\n")
    state = random.getstate()
    result = build(module, mode=mode, lora_wildcard="B", seed=seed)
    assert result[5] == expected
    assert result[0] == [(name + ".safetensors",) for name in expected]
    assert result[1] == result[0]
    assert [len(output) for output in result] == [len(expected)] * 6
    assert random.getstate() == state


def test_missing_fixed_selection_is_not_silently_replaced(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text("first, <lora:A:1>\n")
    with pytest.raises(ValueError, match="missing"):
        build(module, lora_wildcard="missing")
    assert model_calls == []


def test_input_names_and_order_remain_compatible(source):
    module, _ = source
    assert list(module.LoraWildcardSource.INPUT_TYPES()["required"]) == [
        "model", "clip", "category", "lora_wildcard", "mode", "strength", "base_prompt", "seed",
    ]


@pytest.mark.parametrize("tag,expected", [
    ("<lora:A:0.2>", ("A.safetensors", 0.2, 0.2)),
    ("<lora: Base : 0.2 : 0.3>", ("Base.safetensors", 0.2, 0.3)),
    ("<lora:Base:1e-2:2e-2>", ("Base.safetensors", 0.01, 0.02)),
    ("<lora:Base:+0.2:+0.3>", ("Base.safetensors", 0.2, 0.3)),
])
def test_base_loras_keep_original_weights_and_repeated_applications(source, model_calls, tag, expected):
    module, custom = source
    (custom / "styles.txt").write_text("portrait, <lora:A:0.7>\n")
    result = build(module, base_prompt="base, " + tag)
    assert model_calls == [expected, ("A.safetensors", 0.8, 0.8)]
    assert result[0] == [(expected[0], "A.safetensors")]
    assert result[1] == result[0]


def test_prompt_syntax_remains_literal(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text("portrait, <lora:A:0.7>\n")
    prompt = "a {red|blue} vase, __color__"
    result = build(module, base_prompt=prompt)
    assert result[3] == [prompt + ", portrait"]
    assert result[2] == [[prompt + ", portrait"]]


def test_model_resolution_requires_filename_boundary(source, model_calls, monkeypatch):
    import folder_paths
    module, custom = source
    (custom / "styles.txt").write_text("portrait, <lora:A:0.7>\n")
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda kind: ["aaa/NotA.safetensors", "zzz/A.safetensors"])
    build(module)
    assert model_calls == [("zzz/A.safetensors", 0.8, 0.8)]


def test_break_keeps_original_section_punctuation(source, model_calls):
    module, custom = source
    (custom / "styles.txt").write_text("portrait, <lora:A:0.7>\n")
    result = build(module, base_prompt="flowers, BREAK sky")
    assert result[3] == ["flowers, BREAK sky, portrait"]
    assert result[2] == [["flowers,", "sky, portrait"]]


@pytest.mark.parametrize("unavailable", ["missing", "permission"])
def test_unavailable_directory_does_not_break_node_schema(source, monkeypatch, unavailable):
    module, custom = source
    if unavailable == "missing":
        custom.rmdir()
    else:
        def denied(path):
            raise PermissionError(path)
        monkeypatch.setattr(module.os, "listdir", denied)
    schema = module.LoraWildcardSource.INPUT_TYPES()["required"]
    assert schema["category"][0] == ["styles", "poses"]
    assert schema["lora_wildcard"][0] == ["(styles.txt not found)"]
