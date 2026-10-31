"""Run the original Switch tutorial PNGs through Chrome and compare their routes.

Set IMPACT_TUTORIALS_DIR to the ComfyUI-extension-tutorials checkout.
Requires its referenced input images and ControlNet Auxiliary's OpenPose node.
"""
import copy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from e2e_ui import browser_session, options


def output_pixels(ui, entry):
    images = {}
    for node_id, output in ui.images(entry).items():
        assert len(output) == 1
        images[node_id] = (output[0].size, output[0].tobytes())
    return images


def main():
    args = options(__doc__)
    tutorials = Path(os.environ.get("IMPACT_TUTORIALS_DIR",
                                   Path(__file__).resolve().parents[4] / "ComfyUI-extension-tutorials"))
    directory = tutorials / "ComfyUI-Impact-Pack" / "tutorial"
    artifacts = args.artifacts
    for name in ("switch-simple", "inversed-switch-simple", "select_on_execution", "select_on_prompt"):
        args.artifacts = artifacts / name if artifacts else None
        path = directory / f"{name}.png"
        with Image.open(path) as image:
            workflow = json.loads(image.info["workflow"])
        with browser_session(args) as ui:
            ui.page.keyboard.press("Escape")
            ui.new_workflow()
            ui.load_workflow(path, len(workflow["nodes"]))
            with ui.page.expect_request(lambda request: urlsplit(request.url).path in ("/prompt", "/api/prompt")
                                        and request.method == "POST") as submitted:
                initial = output_pixels(ui, ui.run())
            prompt = submitted.value.post_data_json["prompt"]

            if name == "switch-simple":
                sources = ["1", "10", "11"]
                reference = {key: copy.deepcopy(prompt[key]) for key in sources}
                selector, widget = "ImpactSwitch", "select"
                expected_nodes = {value: {"15"} for value in (1, 2, 3)}
            elif name == "inversed-switch-simple":
                sources = ["3", "5", "8"]
                reference = {key: copy.deepcopy(prompt[key]) for key in ["1"] + sources}
                for key in sources:
                    reference[key]["inputs"]["image"] = ["1", 0]
                selector, widget = "ImpactInversedSwitch", "select"
                expected_nodes = {1: {"6"}, 2: {"7"}, 3: {"9"}}
            else:
                sources = ["13", "15"]
                reference = {key: copy.deepcopy(prompt[key]) for key in ["10"] + sources}
                for key in sources:
                    reference[key]["inputs"]["image"] = ["10", 0]
                selector, widget = "PrimitiveNode", "value"
                expected_nodes = ({1: {"33"}, 2: {"33"}} if name == "select_on_execution"
                                  else {1: {"25", "33"}, 2: {"26", "33"}})
            for index, node_id in enumerate(sources, 1):
                reference[f"expected{index}"] = {"class_type": "PreviewImage", "inputs": {"images": [node_id, 0]}}

            ui.new_workflow()
            ui.load_workflow({"name": "switch-reference.json", "mimeType": "application/json",
                              "buffer": json.dumps(reference).encode()}, len(reference))
            expected = output_pixels(ui, ui.run())
            assert len(set(expected.values())) == len(sources), "Reference images must differ"
            assert initial.keys() == expected_nodes[2], f"{name}: initial output nodes differ"
            assert all(pixels == expected["expected2"] for pixels in initial.values())

            ui.new_workflow()
            ui.load_workflow(path, len(workflow["nodes"]))
            for value in [1, 2] + ([3] if len(sources) == 3 else []) + [2, None]:
                if value is None:
                    saved = ui.app.evaluate("app => app.graph.serialize()")
                    ui.new_workflow()
                    ui.load_workflow({"name": f"{name}-saved.json", "mimeType": "application/json",
                                      "buffer": json.dumps(saved).encode()}, len(workflow["nodes"]))
                    value = 2
                else:
                    ui.set_number(selector, widget, value)
                actual = output_pixels(ui, ui.run())
                assert actual.keys() == expected_nodes[value], f"{name}: select={value} output nodes differ"
                assert all(pixels == expected[f"expected{value}"] for pixels in actual.values()), f"{name}: output pixels differ"
            print(f"PASS: {name}, original PNG, all routes, core-node pixel parity, saved JSON import")


if __name__ == "__main__":
    main()
