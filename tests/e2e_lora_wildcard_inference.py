"""Compare LoraWildcardSource with core LoRA/CLIP/sampler nodes in Chrome.

Requires v1-5-pruned-emaonly-fp16.safetensors and SD1.5/etc/{add_detail,
detailmaker}.safetensors. Configure custom_wildcards as wildcards/model_samples.
Uses real models and HTTP execution, with no model or route substitutions.
For exact pixels, start the server with --deterministic --force-fp32 --fp32-vae
--disable-dynamic-vram --use-pytorch-cross-attention.
"""
import hashlib
import json

from e2e_ui import browser_session, options
from lora_source_case import ALIAS_SECTION, BASE, CHECKPOINT, FIRST_LABEL_SECTION, FIRST_SECTION, LAST_SECTIONS, LORAS, SAVED_LABEL, SAVED_SECTION
from test_e2e_widget_catalog import export_workflow, full_zoom, setting, show_node, widget_state


REFERENCE_SECTIONS = [*LAST_SECTIONS, SAVED_SECTION, FIRST_LABEL_SECTION, ALIAS_SECTION]
REFERENCE_NAMES = [*LORAS, SAVED_LABEL, LORAS[0], LORAS[0]]


def workflow():
    graph = {}

    def add(kind, **inputs):
        node_id = str(len(graph) + 1)
        graph[node_id] = {"class_type": kind, "inputs": inputs}
        return node_id

    checkpoint = add("CheckpointLoaderSimple", ckpt_name=CHECKPOINT)
    negative = add("CLIPTextEncode", clip=[checkpoint, 1], text="")
    latent = add("EmptyLatentImage", width=128, height=128, batch_size=1)
    seed = add("PrimitiveInt", value=12345)
    # A linked string bypasses CLIPTextEncode's frontend dynamic-prompt widget.
    # The reference must receive the same literal text as the source node.
    literal = add("PrimitiveString", value=FIRST_SECTION)

    def preview(model, conditioning):
        sampled = add("KSampler", model=model, positive=conditioning, negative=[negative, 0],
                      latent_image=[latent, 0], seed=[seed, 0], steps=4, cfg=7.0,
                      sampler_name="euler", scheduler="normal", denoise=1.0)
        decoded = add("VAEDecode", samples=[sampled, 0], vae=[checkpoint, 2])
        return add("PreviewImage", images=[decoded, 0])

    def encode(clip, last):
        first = add("CLIPTextEncode", clip=clip, text=[literal, 0])
        second = add("CLIPTextEncode", clip=clip, text=last)
        return add("ConditioningConcat", conditioning_to=[first, 0], conditioning_from=[second, 0])

    source = add("LoraWildcardSource", model=[checkpoint, 0], clip=[checkpoint, 1], category="models",
                 lora_wildcard=LORAS[0], mode="batch", strength=0.6, base_prompt=BASE, seed=1)
    actual = preview([source, 0], [source, 2])
    text = add("PreviewAny", source=[source, 3])
    names = add("PreviewAny", source=[source, 5])
    references = []
    for index, lora in enumerate([*LORAS, LORAS[0], LORAS[0], LORAS[0]]):
        base = add("LoraLoader", model=[checkpoint, 0], clip=[checkpoint, 1],
                   lora_name=LORAS[0] + ".safetensors", strength_model=0.1, strength_clip=0.2)
        selected = add("LoraLoader", model=[base, 0], clip=[base, 1],
                       lora_name=lora + ".safetensors", strength_model=0.6, strength_clip=0.6)
        if index == 0:
            selected = add("LoraLoader", model=[selected, 0], clip=[selected, 1],
                           lora_name=LORAS[1] + ".safetensors", strength_model=-0.15, strength_clip=0.2)
        conditioning = encode([selected, 1], REFERENCE_SECTIONS[index])
        references.append(preview([selected, 0], [conditioning, 0]))
    plain = encode([checkpoint, 1], LAST_SECTIONS[0])
    unpatched = preview([checkpoint, 0], [plain, 0])
    return graph, source, actual, text, names, references, unpatched


def main():
    args = options(__doc__)
    assert args.artifacts, "Pass --artifacts to retain the native workflow and results"
    graph, source, actual, text, names, references, unpatched = workflow()
    results = []
    with browser_session(args) as ui:
        ui.page.keyboard.press("Escape")
        ui.new_workflow()
        ui.load_workflow({"name": "lora-inference-api.json", "mimeType": "application/json",
                          "buffer": json.dumps(graph).encode()}, len(graph))
        full_zoom(ui)
        show_node(ui, source)
        ui.choose("LoraWildcardSource", "control_after_generate", "fixed")
        before = widget_state(ui, source)
        saved = args.artifacts / "inference-workflow.json"
        export_workflow(ui, saved)

        def check(label, expected):
            entry = ui.run()
            # Offscreen previews load lazily; fit all outputs into the view.
            ui.page.get_by_role("button", name="Fit View (.)", exact=True).click()
            images = ui.images(entry)
            assert len(images[actual]) == len(expected)
            for image, index in zip(images[actual], expected):
                reference = images[references[index]][0]
                assert image.size == reference.size == (128, 128)
                assert image.tobytes() == reference.tobytes(), f"{label}: item {index} differs from core nodes"
            assert images[references[0]][0].tobytes() != images[references[1]][0].tobytes()
            assert images[references[0]][0].tobytes() != images[unpatched][0].tobytes()
            assert entry["outputs"][text]["text"] == [FIRST_SECTION + " BREAK " + REFERENCE_SECTIONS[i] for i in expected]
            assert entry["outputs"][names]["text"] == [REFERENCE_NAMES[i] for i in expected]
            hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images[actual]]
            for i, image in enumerate(images[actual]):
                image.save(args.artifacts / f"{label}-{i}.png")
            results.append({"case": label, "expected_entries": expected, "pixel_sha256": hashes})
            print(f"PASS: {label}, {len(expected)} output(s), exact core-node pixel parity", flush=True)
            full_zoom(ui)
            show_node(ui, source)

        def check_native(label, expected):
            check(label, expected)
            before = widget_state(ui, source)
            saved = args.artifacts / (label + "-workflow.json")
            export_workflow(ui, saved)
            for restore_by_name in (False, True):
                setting(ui, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
                ui.new_workflow()
                ui.load_workflow(saved, len(graph))
                full_zoom(ui)
                show_node(ui, source)
                assert widget_state(ui, source) == before
                check(label + ("-named" if restore_by_name else "-positional"), expected)

        check("batch", [0, 1])
        ui.new_workflow()
        ui.load_workflow(saved, len(graph))
        full_zoom(ui)
        show_node(ui, source)
        assert widget_state(ui, source) == before
        check("native-reload", [0, 1])
        ui.choose("LoraWildcardSource", "mode", "fixed")
        ui.choose("LoraWildcardSource", "lora_wildcard", LORAS[1])
        check("fixed", [1])
        ui.choose("LoraWildcardSource", "mode", "sequential")
        check("sequential", [1])
        ui.choose("LoraWildcardSource", "mode", "random")
        check("random", [0])

        ui.choose("LoraWildcardSource", "mode", "fixed")
        ui.choose("LoraWildcardSource", "category", "labels")
        ui.page.wait_for_function("""({app, value}) => app.graph._nodes.find(n => n.type === 'LoraWildcardSource')
            .widgets.find(w => w.name === 'lora_wildcard').options.values.includes(value)""",
            arg={"app": ui.app, "value": SAVED_LABEL})
        ui.choose("LoraWildcardSource", "lora_wildcard", SAVED_LABEL)
        for mode, seed, expected in [("fixed", 1, [2]), ("random", 5, [2]),
                                     ("sequential", 2, [3]), ("batch", 0, [3, 2])]:
            ui.choose("LoraWildcardSource", "mode", mode)
            ui.set_number("LoraWildcardSource", "seed", seed)
            label = "saved-label" if mode == "fixed" else "mixed-" + mode
            check_native(label, expected)

        # An old API workflow may contain an alias absent from today's combo
        # options. Import it through the UI, then use native Export/reload.
        graph[source]["inputs"]["category"] = "pose"
        ui.new_workflow()
        ui.load_workflow({"name": "legacy-alias-api.json", "mimeType": "application/json",
                          "buffer": json.dumps(graph).encode()}, len(graph))
        full_zoom(ui)
        show_node(ui, source)
        ui.choose("LoraWildcardSource", "control_after_generate", "fixed")
        assert widget_state(ui, source)["widgets"][0]["value"] == "pose"
        check_native("legacy-alias", [4])
    (args.artifacts / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
