"""Real ComfyUI tensor/validator checks; run separately from wildcard stubs.

IMPACT_COMFY_ROOT=/path/to/ComfyUI python -m pytest tests/test_image_list_batch.py -v
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    comfy_root = os.environ.get("IMPACT_COMFY_ROOT")
    if not comfy_root:
        pytest.fail("Set IMPACT_COMFY_ROOT to a ComfyUI checkout with its dependencies installed", pytrace=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(comfy_root)
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        import execution
        import nodes
        from impact import util_nodes
        assert Path(util_nodes.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
        yield SimpleNamespace(node=util_nodes.ImageListToImageBatch(), util=util_nodes,
                              execution=execution, nodes=nodes)


@pytest.mark.parametrize("batch_sizes, unbatched", [
    ((1,) * 4, ()), ((1,) * 16, ()), ((1,) * 64, ()), ((2, 1, 3), ()),
    ((1,) * 4, range(4)), ((1,) * 16, range(16)), ((1,) * 64, range(64)),
    ((1, 2, 3), (0,)), ((2, 1, 3), (1,)),
], ids=["4-frames", "16-frames", "64-frames", "unequal-batches",
        "3d-4-frames", "3d-16-frames", "3d-64-frames", "3d-first", "3d-later"])
def test_uniform_frames_allocate_at_most_one_concatenated_batch(runtime, batch_sizes, unbatched):
    count = sum(batch_sizes)
    expected = torch.arange(count).div(64).reshape(count, 1, 1, 1).expand(count, 8, 12, 3).clone()
    frames = [frame[0] if index in unbatched else frame
              for index, frame in enumerate(expected.split(batch_sizes))]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        result = runtime.node.doit(frames)[0]
    allocated_bytes = sum(event.cpu_memory_usage for event in profile.events() if event.name == "aten::cat")
    batch_bytes = count * 8 * 12 * 3 * frames[0].element_size()
    assert allocated_bytes > 0, "No concatenation allocations recorded; check profiler event/metric support"
    assert allocated_bytes <= batch_bytes, f"Concatenation allocated {allocated_bytes} bytes; one batch is {batch_bytes}"
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "expanded"])
def test_pixels_order_sub_batches_and_inputs_are_preserved(runtime, dtype, layout):
    expected = torch.arange(6 * 4 * 8 * 3, dtype=dtype).reshape(6, 4, 8, 3) / 1024
    if layout == "transposed":
        expected = expected.transpose(1, 2)
    elif layout == "expanded":
        expected = expected[:, :1, :1, :].expand(6, 4, 8, 3)
    images = list(expected.split([2, 1, 3]))
    before = [image.clone() for image in images]
    result = runtime.node.doit(images)[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for image, original in zip(images, before):
        torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_empty_list_blocks_downstream_and_nonempty_input_recovers(runtime):
    async def execute():
        outputs, ui, expanded, pending = await runtime.execution.get_output_data(
            "empty-batch", "batch", runtime.node, {"images": []})
        assert len(outputs) == len(runtime.node.RETURN_TYPES) == 1
        assert len(outputs[0]) == 1
        assert isinstance(outputs[0][0], runtime.execution.ExecutionBlocker)
        assert outputs[0][0].message is None
        assert ui == {} and not expanded and not pending

        scale = runtime.nodes.ImageScale()
        inputs = {"image": outputs[0], "upscale_method": ["nearest-exact"],
                  "width": [12], "height": [8], "crop": ["disabled"]}
        scaled, ui, expanded, pending = await runtime.execution.get_output_data(
            "empty-batch", "scale", scale, inputs)
        assert isinstance(scaled[0][0], runtime.execution.ExecutionBlocker)
        assert ui == {} and not expanded and not pending

        image = torch.full((1, 4, 6, 3), 0.25)
        outputs, _, _, _ = await runtime.execution.get_output_data(
            "filled-batch", "batch", runtime.node, {"images": [image]})
        inputs["image"] = outputs[0]
        scaled, _, _, _ = await runtime.execution.get_output_data(
            "filled-batch", "scale", scale, inputs)
        torch.testing.assert_close(scaled[0][0], torch.full((1, 8, 12, 3), 0.25), rtol=0, atol=0)

    asyncio.run(execute())


@pytest.mark.parametrize("batched", [False, True])
def test_single_image_keeps_storage_and_batch_dimension(runtime, batched):
    image = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    expected = image.unsqueeze(0)
    supplied = expected if batched else image
    result = runtime.node.doit([supplied])[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.data_ptr() == supplied.data_ptr()
    if batched:
        assert result is supplied


@pytest.mark.parametrize("batched", [(False, False), (False, True), (True, False), (True, True)],
                         ids=["HWC-HWC", "HWC-BHWC", "BHWC-HWC", "BHWC-BHWC"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is needed for device compatibility"))])
def test_3d_and_4d_inputs_preserve_batch_axis(runtime, batched, device):
    counts = [2 if batched[0] else 1, 3 if batched[1] else 1]
    expected = torch.arange(sum(counts) * 4 * 8 * 3, dtype=torch.float32, device=device).reshape(sum(counts), 4, 8, 3) / 1024
    images = [image if batch else image[0] for image, batch in zip(expected.split(counts), batched)]
    before = [image.clone() for image in images]
    result = runtime.node.doit(images)[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for image, original in zip(images, before):
        torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_empty_sub_batches_do_not_change_order(runtime):
    image = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    result = runtime.node.doit([image[:0], image, image[:0]])[0]
    torch.testing.assert_close(result, image, rtol=0, atol=0)


def test_nonfinite_values_and_signed_zero_keep_their_bits(runtime):
    expected = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"), 0.5]).reshape(2, 1, 1, 3)
    result = runtime.node.doit(list(expected.split(1)))[0]
    assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("batched", [(False, False), (False, True), (True, False), (True, True)],
                         ids=["HWC-HWC", "HWC-BHWC", "BHWC-HWC", "BHWC-BHWC"])
def test_mixed_sizes_keep_center_crop_and_resize(runtime, batched):
    first = torch.tensor([1.0, 0.0, 0.0]).expand(1, 4, 4, 3).clone()
    wide = torch.ones(1, 4, 8, 3)
    wide[:, :, 2:6] = torch.tensor([0.0, 1.0, 0.0])
    tall = torch.ones(1, 8, 4, 3)
    tall[:, 2:6, :] = torch.tensor([0.0, 0.0, 1.0])
    large = torch.zeros(1, 8, 8, 3)
    large[:, 2:6, 2:6] = 1.0
    images = [first, wide, tall, large]
    ranks = (batched[0],) + (batched[1],) * 3
    counts = (2 if batched[0] else 1,) + (3 if batched[1] else 1,) * 3
    images = [image.repeat(count, 1, 1, 1) if batch else image[0]
              for image, batch, count in zip(images, ranks, counts)]
    before = [image.clone() for image in images]
    result = runtime.node.doit(images)[0]
    expected = torch.zeros(4, 4, 4, 3)
    expected[:3] = torch.eye(3).reshape(3, 1, 1, 3)
    # Main eace4a9 output with ComfyUI 61e524b; fixed pixels distinguish resize kernels.
    expected[3] = torch.tensor([[1, 13, 13, 1], [12, 230, 230, 12],
                                [12, 230, 230, 12], [1, 13, 13, 1]], dtype=torch.float32).unsqueeze(-1) / 255
    expected = expected.repeat_interleave(torch.tensor(counts), dim=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for image, original in zip(images, before):
        torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_mixed_channels_keep_progressive_truncation(runtime):
    images = [torch.tensor([0.25, 0.5, 0.75, 1.0]).expand(2, 4, 8, 4).clone(),
              torch.tensor([0.5, 0.75, 1.0]).expand(1, 4, 8, 3).clone(),
              torch.full((1, 4, 8, 1), 0.75)]
    before = [image.clone() for image in images]
    result = runtime.node.doit(images)[0]
    expected = torch.tensor([0.25, 0.25, 0.5, 0.75]).reshape(4, 1, 1, 1).expand(4, 4, 8, 1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for image, original in zip(images, before):
        torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_mixed_dtypes_keep_sequential_promotion_rounding(runtime):
    images = [torch.zeros(1, 2, 4, 3, dtype=torch.float16),
              torch.full((1, 2, 4, 3), 2049, dtype=torch.int64),
              torch.ones(1, 2, 4, 3, dtype=torch.float32)]
    result = runtime.node.doit(images)[0]
    expected = torch.tensor([0.0, 2048.0, 1.0]).reshape(3, 1, 1, 1).expand(3, 2, 4, 3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is needed for device compatibility")
@pytest.mark.parametrize("devices", [("cuda", "cpu"), ("cpu", "cuda")])
def test_output_stays_on_first_device(runtime, devices):
    images = [torch.full((1, 2, 4, 3), value, device=device)
              for value, device in zip([0.25, 0.75], devices)]
    result = runtime.node.doit(images)[0]
    expected = torch.tensor([0.25, 0.75], device=images[0].device).reshape(2, 1, 1, 1).expand(2, 2, 4, 3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert [str(image.device).split(":")[0] for image in images] == list(devices)


def test_existing_api_prompt_and_list_execution_need_no_new_input(runtime, monkeypatch):
    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "ImageListToImageBatch", type(runtime.node))
    monkeypatch.setitem(runtime.nodes.NODE_CLASS_MAPPINGS, "ImageBatchToImageList", runtime.util.ImageBatchToImageList)
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 16, "height": 16, "batch_size": 2, "color": 0}},
        "2": {"class_type": "ImageBatchToImageList", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "ImageListToImageBatch", "inputs": {"images": ["2", 0]}},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
    }
    valid, error, outputs, node_errors = asyncio.run(runtime.execution.validate_prompt("batch-compat", prompt, None))
    assert valid and error is None and outputs == ["4"] and node_errors == {}
    images = [torch.zeros(1, 2, 4, 3), torch.ones(1, 2, 4, 3)]
    results = asyncio.run(runtime.execution._async_map_node_over_list(
        "batch-compat", "3", runtime.node, {"images": images}, "doit"))
    torch.testing.assert_close(results[0][0], torch.stack([images[0][0], images[1][0]]), rtol=0, atol=0)
