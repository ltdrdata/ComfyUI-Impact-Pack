import io
import json

import pytest
from PIL import Image

from test_e2e_preview_bridge import add_node, color, connect, file_bytes, number, ui as ui
from test_e2e_widget_catalog import export_workflow, setting


RED, BLUE = (255, 0, 0), (0, 0, 255)


def colors(browser, entry, node_id):
    found = []
    for meta in entry["outputs"][node_id]["images"]:
        with Image.open(io.BytesIO(file_bytes(browser, meta))) as image:
            assert image.size == (64, 64)
            (_, only), = image.convert("RGB").getcolors()
            found.append(only)
    return found


@pytest.mark.parametrize("restore_by_name", [False, True], ids=["positional", "named"])
def test_each_image_of_a_list_gets_the_segs_of_its_own_place(ui, restore_by_name):
    browser, args = ui
    browser.page.set_viewport_size({"width": 2700, "height": 1300})
    setting(browser, "Restore widget values", "Comfy.Workflow.NamedValuesRestore", restore_by_name)
    pictures = []
    for row, value in enumerate((0xFF0000, 0x0000FF)):
        picture = add_node(browser, "EmptyImage", 100, 100 + 300 * row)
        number(browser, picture, "width", 64)
        number(browser, picture, "height", 64)
        color(browser, picture, value)
        pictures.append(picture)
    image_list = add_node(browser, "ImpactMakeImageList", 520, 100)
    connect(browser, pictures[0], image_list)
    connect(browser, pictures[1], image_list, target_slot=1)
    solid = add_node(browser, "SolidMask", 100, 720)
    number(browser, solid, "width", 64)
    number(browser, solid, "height", 64)
    to_segs = add_node(browser, "MaskToSEGS", 520, 720)
    connect(browser, solid, to_segs)
    with_picture = []
    for row, picture in enumerate(reversed(pictures)):
        node = add_node(browser, "SetDefaultImageForSEGS", 900, 380 + 190 * row)
        connect(browser, to_segs, node)
        connect(browser, picture, node, target_slot="image")
        with_picture.append(node)
    segs_list = add_node(browser, "ImpactMakeAnyList", 1260, 520)
    connect(browser, with_picture[0], segs_list)
    connect(browser, with_picture[1], segs_list, target_slot=1)
    previews = {}
    for row, (taken, together) in enumerate((("ImpactDecomposeSEGSList", "ImpactAssembleSEGSList"), ("ImpactDecomposeSEGS", "ImpactAssembleSEGS"))):
        decompose = add_node(browser, taken, 1520, 250 + 560 * row)
        connect(browser, segs_list, decompose)
        assemble = add_node(browser, together, 1800, 250 + 560 * row)
        connect(browser, decompose, assemble)
        connect(browser, decompose, assemble, source_slot=1, target_slot=1)
        paste = add_node(browser, "SEGSPaste", 2040, 100 + 560 * row)
        connect(browser, image_list, paste)
        connect(browser, assemble, paste, target_slot="segs")
        number(browser, paste, "feather", 0)
        previews[together] = add_node(browser, "PreviewImage", 2350, 100 + 560 * row)
        connect(browser, paste, previews[together])

    response = browser.page.request.get(f"{browser.base_url}/object_info")
    assert response.ok
    definitions = response.json()
    for name in ("ImpactDecomposeSEGS", "ImpactDecomposeSEGSList"):
        assert definitions[name]["input"]["required"] == {"segs": ["SEGS"]}
        assert definitions[name]["output"] == ["SEGS_HEADER", "SEG_ELT"]
    assert definitions["ImpactAssembleSEGS"]["input"] == {"required": {"seg_header": ["SEGS_HEADER"], "seg_elt": ["SEG_ELT", {"lazy": True}]}}
    assert definitions["ImpactAssembleSEGSList"]["input"] == {"required": {"seg_header": ["SEGS_HEADER"], "seg_elt": ["SEG_ELT"]}}
    for name in ("ImpactAssembleSEGS", "ImpactAssembleSEGSList"):
        assert definitions[name]["output"] == ["SEGS"]
    assert definitions["ImpactDecomposeSEGS"]["output_is_list"] == [False, True]
    assert definitions["ImpactDecomposeSEGSList"]["output_is_list"] == [True, True]
    assert definitions["ImpactAssembleSEGS"]["output_is_list"] == [False]
    assert definitions["ImpactAssembleSEGSList"]["output_is_list"] == [True]

    workflow = args.artifacts / "native.json"
    for phase in ("built", "reloaded"):
        if phase == "reloaded":
            export_workflow(browser, workflow)
            saved = json.loads(workflow.read_text())
            browser.new_workflow()
            browser.load_workflow(workflow, len(saved["nodes"]))
            restored = browser.app.evaluate("app => app.graph.serialize()")
            assert [(node["id"], node["type"]) for node in restored["nodes"]] == [(node["id"], node["type"]) for node in saved["nodes"]]
            assert sorted(map(list, restored["links"])) == sorted(saved["links"])
        entry = browser.run()
        browser.images(entry)
        assert colors(browser, entry, previews["ImpactAssembleSEGSList"]) == [BLUE, RED], phase
        assert colors(browser, entry, previews["ImpactAssembleSEGS"]) == [RED, RED], phase
