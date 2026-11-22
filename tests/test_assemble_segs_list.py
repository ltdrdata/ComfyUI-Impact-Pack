import asyncio
import copy
import itertools
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


WARNING = ("[Impact Pack] Assemble (SEGS list): an element is of none of the given headers, so all elements go into "
           "one SEGS. Take the headers and the elements from Decompose (SEGS list).")


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
        from aiohttp import web
        from server import PromptServer
        patch.setattr(PromptServer, "instance", SimpleNamespace(
            routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None), raising=False)
        from impact import segs_nodes
        assert Path(segs_nodes.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
        yield SimpleNamespace(execution=execution, nodes=segs_nodes, SEG=segs_nodes.SEG)


def run(runtime, node, **inputs):
    outputs, _, _, _ = asyncio.run(runtime.execution.get_output_data("prompt", "1", node(), inputs))
    return outputs


def decompose(runtime, listed, node_id="7"):
    return run(runtime, runtime.nodes.DecomposeSEGSList, segs=listed, unique_id=[node_id])


def assemble(runtime, headers, elements):
    return run(runtime, runtime.nodes.AssembleSEGSList, seg_header=headers, seg_elt=elements)[0]


def make(runtime, sizes, counts):
    labels = itertools.count()
    listed = []
    for size, count in zip(sizes, counts):
        elements = []
        for _ in range(count):
            label = next(labels)
            mask = np.full((4, 4), (label + 1) / 16, dtype=np.float32)
            picture = torch.full((1, 4, 4, 3), (label + 1) / 32)
            elements.append(runtime.SEG(picture, mask, 0.5 + label / 64, (0, 0, 4, 4), (1, 1, 3, 3), f"label {label}", object()))
        listed.append((size, elements))
    return listed


def assert_same(actual, expected):
    assert [tuple(segs[0]) for segs in actual] == [tuple(segs[0]) for segs in expected]
    assert [[seg.label for seg in segs[1]] for segs in actual] == [[seg.label for seg in segs[1]] for segs in expected]
    for found, wanted in zip(itertools.chain(*(segs[1] for segs in actual)), itertools.chain(*(segs[1] for segs in expected))):
        assert found.cropped_image is wanted.cropped_image
        assert found.cropped_mask is wanted.cropped_mask
        assert found.control_net_wrapper is wanted.control_net_wrapper
        assert found[2:6] == wanted[2:6]


SHAPES = [(1,), (0,), (0, 2), (2, 0), (0, 0), (4, 1, 0, 2, 1), ()]


@pytest.mark.parametrize("sizes,counts", [("one size", counts) for counts in SHAPES] + [("sizes differ", counts) for counts in SHAPES[:-1]], ids=str)
def test_a_list_of_segs_comes_back_as_that_list(runtime, counts, sizes):
    listed = make(runtime, [(768, 768) if sizes == "one size" else (64 + index, 96 + 2 * index) for index in range(len(counts))], counts)
    headers, elements = decompose(runtime, listed)
    assert_same(assemble(runtime, headers, elements), listed)


@pytest.mark.parametrize("counts", SHAPES[:-1], ids=str)
def test_decompose_segs_list_returns_what_decompose_segs_returns(runtime, counts):
    listed = make(runtime, [(64 + index, 96) for index in range(len(counts))], counts)
    plain_headers, plain_elements = run(runtime, runtime.nodes.DecomposeSEGS, segs=listed)
    headers, elements = decompose(runtime, listed)
    assert headers == plain_headers and all(isinstance(header, tuple) for header in headers)
    assert [hash(header) for header in headers] == [hash(header) for header in plain_headers]
    assert all(isinstance(seg, runtime.SEG) for seg in elements)
    assert [seg._fields for seg in elements] == [seg._fields for seg in plain_elements]
    assert_same([((0, 0), elements)], [((0, 0), plain_elements)])


@pytest.mark.parametrize("counts", SHAPES[:-1], ids=str)
def test_decompose_segs_and_assemble_segs_are_as_they_were(runtime, counts):
    listed = make(runtime, [(64 + index, 96) for index in range(len(counts))], counts)
    wanted = [seg for segs in listed for seg in segs[1]]
    headers, elements = run(runtime, runtime.nodes.DecomposeSEGS, segs=listed)
    assert all(found is given[0] for found, given in zip(headers, listed))
    assert len(elements) == len(wanted) and all(found is given for found, given in zip(elements, wanted))
    assembled, = run(runtime, runtime.nodes.AssembleSEGS, seg_header=headers, seg_elt=elements)
    assert_same(assembled, [(listed[0][0], wanted)])


@pytest.mark.parametrize("counts", SHAPES[:-1], ids=str)
def test_assemble_segs_puts_what_decompose_segs_list_took_apart_into_one_segs(runtime, counts):
    listed = make(runtime, [(64 + index, 96) for index in range(len(counts))], counts)
    headers, elements = decompose(runtime, listed)
    assembled, = run(runtime, runtime.nodes.AssembleSEGS, seg_header=headers, seg_elt=elements)
    assert_same(assembled, [(listed[0][0], [seg for segs in listed for seg in segs[1]])])


def test_the_same_segs_given_twice_stays_two_segs(runtime):
    segs, = make(runtime, [(768, 768)], (2,))
    headers, elements = decompose(runtime, [segs, segs, segs])
    assert_same(assemble(runtime, headers, elements), [segs, segs, segs])


def test_two_decompose_nodes_into_one_assemble_keep_their_segs_apart(runtime):
    listed = make(runtime, [(64, 64), (64, 64), (64, 64)], (2, 1, 2))
    first = decompose(runtime, listed[:2], "7")
    second = decompose(runtime, listed[2:], "8")
    assert_same(assemble(runtime, first[0] + second[0], second[1] + first[1]), listed)


@pytest.mark.parametrize("node, inputs, changed", [
    ("Edit_SEG_ELT", {"label_opt": ["edited"]}, "label"),
    ("Edit_SEG_ELT", {"bbox_opt": [(0, 0, 2, 2)]}, "bbox"),
    ("SEG_ELT_BBOX_ScaleBy", {"scale_by": [2.0]}, "bbox"),
], ids=["edit label", "edit bbox", "scale bbox"])
def test_an_edited_element_returns_to_its_own_segs(runtime, node, inputs, changed):
    listed = make(runtime, [(64, 64), (96, 96), (64, 64)], (2, 0, 3))
    headers, elements = decompose(runtime, listed)
    name = "seg" if node == "SEG_ELT_BBOX_ScaleBy" else "seg_elt"
    edited, = run(runtime, getattr(runtime.nodes, node), **{name: elements}, **inputs)
    assert len(edited) == len(elements)
    for after, before in zip(edited, elements):
        assert getattr(after, changed) != getattr(before, changed)
        assert np.array_equal(np.asarray(after.cropped_image), np.asarray(before.cropped_image))
        assert after.control_net_wrapper is before.control_net_wrapper and after.confidence == before.confidence
    assembled = assemble(runtime, headers, edited)
    assert [tuple(segs[0]) for segs in assembled] == [(64, 64), (96, 96), (64, 64)]
    assert [[seg.confidence for seg in segs[1]] for segs in assembled] == [[seg.confidence for seg in segs[1]] for segs in listed]


def test_an_edited_element_that_was_not_decomposed_stays_a_plain_seg(runtime):
    element = make(runtime, [(64, 64)], (1,))[0][1]
    edited, = run(runtime, runtime.nodes.Edit_SEG_ELT, seg_elt=element, label_opt=["edited"])
    scaled, = run(runtime, runtime.nodes.SEG_ELT_BBOX_ScaleBy, seg=element, scale_by=[2.0])
    assert type(edited[0]) is runtime.SEG and type(scaled[0]) is runtime.SEG


@pytest.mark.parametrize("how", ["From SEG_ELT", "_replace", "deepcopy"])
def test_an_element_read_or_copied_returns_to_its_own_segs(runtime, how):
    listed = make(runtime, [(64, 64), (96, 96)], (1, 2))
    headers, elements = decompose(runtime, listed)
    if how == "From SEG_ELT":
        passed = run(runtime, runtime.nodes.From_SEG_ELT, seg_elt=elements)[0]
    elif how == "_replace":
        passed = [seg._replace(confidence=seg.confidence / 2) for seg in elements]
        assert [seg.confidence for seg in passed] == [seg.confidence / 2 for seg in elements]
    else:
        passed = copy.deepcopy(elements)
        headers = copy.deepcopy(headers)
    assembled = assemble(runtime, headers, passed)
    assert [tuple(segs[0]) for segs in assembled] == [(64, 64), (96, 96)]
    assert [[seg.label for seg in segs[1]] for segs in assembled] == [["label 0"], ["label 1", "label 2"]]


@pytest.mark.parametrize("kept", [
    (0, 2, 3, 5), (1,), (2, 3, 4, 5), (), (5, 0, 3),
], ids=["some of each", "one", "a whole SEGS removed", "all removed", "order changed"])
def test_elements_left_after_a_filter_return_to_their_own_segs(runtime, kept):
    listed = make(runtime, [(64, 64), (96, 96), (128, 128)], (2, 1, 3))
    headers, elements = decompose(runtime, listed)
    assembled = assemble(runtime, headers, [elements[index] for index in kept])
    assert [tuple(segs[0]) for segs in assembled] == [(64, 64), (96, 96), (128, 128)]
    labels = [[f"label {index}" for index in kept if index in group] for group in ((0, 1), (2,), (3, 4, 5))]
    assert [[seg.label for seg in segs[1]] for segs in assembled] == labels


@pytest.mark.parametrize("held", ["elements", "headers"])
def test_elements_return_to_their_segs_when_the_decompose_node_has_run_again(runtime, held):
    """A cache can drop the result of Decompose (SEGS list) and keep that of a node after it."""
    listed = make(runtime, [(64, 64), (96, 96), (64, 64)], (2, 1, 3))
    before = decompose(runtime, listed)
    after = decompose(runtime, listed)
    headers, elements = (after[0], before[1]) if held == "elements" else (before[0], after[1])
    assert all(new is not old for new, old in zip(after[0] + after[1], before[0] + before[1]))
    assert_same(assemble(runtime, headers, [elements[index] for index in (5, 0, 2, 4)]),
                [(size, [segs[index] for index in kept]) for (size, segs), kept in zip(listed, ((0,), (0,), (2, 1)))])


def test_segs_given_twice_to_assemble_do_not_share_their_elements(runtime):
    listed = make(runtime, [(64, 64)], (2,))
    headers, elements = decompose(runtime, listed)
    assembled = assemble(runtime, headers + headers, elements)
    assert_same(assembled, listed + listed)
    assert assembled[0][1] is not assembled[1][1]


@pytest.mark.parametrize("case", ["an element made elsewhere", "headers of another node",
                                  "headers of Decompose (SEGS)", "both of Decompose (SEGS)", "a header left out"])
def test_elements_of_no_given_header_go_into_one_segs_with_a_warning(runtime, caplog, case):
    listed = make(runtime, [(64, 64), (96, 96)], (2, 1))
    headers, elements = decompose(runtime, listed)
    if case == "an element made elsewhere":
        elements = elements + make(runtime, [(64, 64)], (1,))[0][1]
    elif case == "headers of another node":
        headers = decompose(runtime, listed, "8")[0]
    elif case == "headers of Decompose (SEGS)":
        headers = run(runtime, runtime.nodes.DecomposeSEGS, segs=listed)[0]
    elif case == "both of Decompose (SEGS)":
        headers, elements = run(runtime, runtime.nodes.DecomposeSEGS, segs=listed)
    else:
        headers = headers[:1]
    with caplog.at_level(logging.WARNING):
        assembled = assemble(runtime, headers, elements)
    assert len(assembled) == 1 and tuple(assembled[0][0]) == (64, 64)
    assert [seg.label for seg in assembled[0][1]] == [seg.label for seg in elements]
    assert [record.getMessage() for record in caplog.records] == [WARNING]


@pytest.mark.parametrize("plain_headers", [False, True], ids=["Decompose (SEGS list)", "headers of Decompose (SEGS)"])
def test_no_warning_when_every_element_has_its_header(runtime, caplog, plain_headers):
    listed = make(runtime, [(64, 64), (96, 96)], (0, 0) if plain_headers else (2, 1))
    headers, elements = run(runtime, runtime.nodes.DecomposeSEGS, segs=listed) if plain_headers else decompose(runtime, listed)
    with caplog.at_level(logging.WARNING):
        assert_same(assemble(runtime, headers, elements), listed)
    assert caplog.records == []


def test_decomposing_again_after_assembling_gives_the_same_list(runtime):
    listed = make(runtime, [(64, 64), (96, 96), (64, 64)], (2, 0, 1))
    for node_id in ("7", "8"):
        headers, elements = decompose(runtime, listed, node_id)
        assembled = assemble(runtime, headers, elements)
        assert_same(assembled, listed)
        listed = assembled


def test_each_image_of_a_list_gets_the_segs_of_its_own_place(runtime):
    images = [torch.full((1, 32, 32, 3), value) for value in (0.25, 0.5, 0.75)]
    pasted = [torch.full((1, 16, 16, 3), value) for value in (1.0, 0.0, 0.125)]
    mask = np.ones((16, 16), dtype=np.float32)
    listed = [((32, 32), [runtime.SEG(crop, mask, 1.0, (0, 0, 16, 16), (0, 0, 16, 16), "square", None)]) for crop in pasted]
    corner = {}
    for name, taken, node in (("list", decompose(runtime, listed), "AssembleSEGSList"),
                              ("one", run(runtime, runtime.nodes.DecomposeSEGS, segs=listed), "AssembleSEGS")):
        assembled, = run(runtime, getattr(runtime.nodes, node), seg_header=taken[0], seg_elt=taken[1])
        results, = run(runtime, runtime.nodes.SEGSPaste, image=images, segs=assembled, feather=[0], alpha=[255])
        assert [float(result[0, 24, 24, 0]) for result in results] == [0.25, 0.5, 0.75]
        corner[name] = [float(result[0, 8, 8, 0]) for result in results]
    assert corner == {"list": [1.0, 0.0, 0.125], "one": [0.125, 0.125, 0.125]}


@pytest.mark.parametrize("dilation, area", [(2, 16), (-2, 4)], ids=["dilate", "erode"])
def test_dilate_mask_seg_elt_runs_and_changes_only_the_mask(runtime, dilation, area):
    mask = np.zeros((9, 9), dtype=np.float32)
    mask[3:6, 3:6] = 1
    picture, wrapper = torch.rand(1, 9, 9, 3), object()
    seg = runtime.SEG(picture, mask, 0.75, (0, 0, 9, 9), (3, 3, 6, 6), "square", wrapper)
    dilated, = run(runtime, runtime.nodes.Dilate_SEG_ELT, seg_elt=[seg], dilation=[dilation])
    assert type(dilated[0]) is runtime.SEG
    assert dilated[0].cropped_image is picture and dilated[0].control_net_wrapper is wrapper and dilated[0][2:6] == seg[2:6]
    assert int(dilated[0].cropped_mask.sum()) == area


def test_a_dilated_element_returns_to_its_own_segs(runtime):
    listed = make(runtime, [(64, 64), (96, 96), (64, 64)], (2, 0, 3))
    headers, elements = decompose(runtime, listed)
    dilated, = run(runtime, runtime.nodes.Dilate_SEG_ELT, seg_elt=elements, dilation=[2])
    assembled = assemble(runtime, headers, dilated)
    assert [tuple(segs[0]) for segs in assembled] == [(64, 64), (96, 96), (64, 64)]
    assert [[seg.label for seg in segs[1]] for segs in assembled] == [[seg.label for seg in segs[1]] for segs in listed]
