import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import multiprocessing
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

from aiohttp import web
import numpy as np
from PIL import Image
import pytest
import requests


PROMPTS = {"positive_points": [[1, 0]], "negative_points": [[0, 1]], "threshold": 0.7}


def image_bytes():
    output = io.BytesIO()
    Image.fromarray(np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8)).convert("RGB").save(output, format="PNG")
    return output.getvalue()


def serve_sam(connection, directory):
    from server import PromptServer
    PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None)
    from impact import impact_server, sam_editor

    state = {"mode": "normal", "active": 0, "max_active": 0, "loaded": 0, "offloaded": 0, "arrivals": 0}
    gate = threading.Event()
    gate.set()
    predicting = threading.Event()
    arrived = asyncio.Event()

    class Model:
        def to(self, device):
            if device == "cpu":
                state["offloaded"] += 1

        def cpu(self):
            self.to("cpu")

    class Predictor:
        def __init__(self, model):
            self.model = model
            self.image = np.array(Image.open(io.BytesIO(image_bytes())))

        def set_image(self, image, color):
            self.image = image.copy()

        def predict(self, point_coords, point_labels, box):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            predicting.set()
            try:
                if not gate.wait(20):
                    raise TimeoutError("Test inference gate was not released")
                if state["mode"] == "inference-failure":
                    raise RuntimeError("Test inference failed")
                return np.array([self.image[:, :, 0] > 127]), np.array([0.9]), None
            finally:
                state["active"] -= 1

    def load_model(checkpoint):
        state["loaded"] += 1
        return Model()

    impact_server.sam_predictor = Predictor(Model())
    impact_server.SamPredictor = sam_editor.SamPredictor = Predictor
    sam_editor.sam_model_registry["vit_b"] = load_model
    impact_server.impact.config.get_config = lambda: {"sam_editor_cpu": True, "sam_editor_model": "sam_vit_b.pth"}
    impact_server.folder_paths.get_full_path = lambda kind, name: None if state["mode"] == "missing-model" else "sam_vit_b.pth"
    impact_server.folder_paths.set_input_directory(str(directory))
    (Path(directory) / "source.png").write_bytes(image_bytes())

    @web.middleware
    async def requests_arrived(request, handler):
        if request.path.startswith(("/sam/", "/impact/sam/")):
            state["arrivals"] += 1
            arrived.set()
        return await handler(request)

    async def control(request):
        data = await request.json()
        state["mode"] = data.get("mode", "normal")
        if data.get("blocked"):
            predicting.clear()
            gate.clear()
        else:
            gate.set()
        return web.json_response(state)

    async def status(request):
        target = int(request.query.get("arrivals", 0))
        while state["arrivals"] < target:
            arrived.clear()
            await arrived.wait()
        if request.query.get("predicting"):
            assert await asyncio.to_thread(predicting.wait, 5)
        return web.json_response(state)

    async def run():
        app = web.Application(middlewares=[requests_arrived])
        app.router.add_post("/impact/sam/detect", impact_server.sam_detect_snapshot)
        app.router.add_post("/sam/detect", impact_server.sam_detect)
        app.router.add_post("/sam/prepare", impact_server.sam_prepare)
        app.router.add_post("/sam/release", impact_server.release_sam)
        app.router.add_post("/test/control", control)
        app.router.add_get("/test/status", status)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        connection.send(site._server.sockets[0].getsockname()[1])
        await asyncio.Event().wait()

    asyncio.run(run())


@pytest.fixture
def sam_http(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=serve_sam, args=(child, tmp_path))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "Isolated SAM HTTP server did not start"
        yield f"http://127.0.0.1:{parent.recv()}"
    finally:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def detect(url, image=None, prompts=None):
    return requests.post(url + "/impact/sam/detect", files={"image": ("snapshot.png", image_bytes() if image is None else image)},
                         data={"prompts": json.dumps(PROMPTS) if prompts is None else prompts}, timeout=10)


def assert_mask(response):
    assert response.status_code == 200, response.text
    assert response.headers["Content-Type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        np.testing.assert_array_equal(np.asarray(image.convert("L")), [[0, 255, 0], [255, 0, 0]])


def test_slow_legacy_request_bodies_do_not_block_other_http_requests(sam_http):
    body = json.dumps(PROMPTS).encode()
    port = int(sam_http.rsplit(":", 1)[1])
    header = (f"POST /sam/detect HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=2) as first, socket.create_connection(("127.0.0.1", port), timeout=2) as second:
        first.sendall(header + body[:1])
        assert requests.get(sam_http + "/test/status?arrivals=1", timeout=2).ok
        second.sendall(header + body[:1])
        assert requests.get(sam_http + "/test/status?arrivals=2", timeout=2).ok
        first.sendall(body[1:])
        second.sendall(body[1:])
        assert b"200 OK" in first.recv(4096)
        assert b"200 OK" in second.recv(4096)


def test_inference_and_legacy_lifecycle_leave_http_responsive(sam_http):
    requests.post(sam_http + "/test/control", json={"blocked": True}, timeout=2).raise_for_status()
    with ThreadPoolExecutor(max_workers=4) as executor:
        detection = executor.submit(detect, sam_http)
        assert requests.get(sam_http + "/test/status?predicting=1", timeout=3).ok
        prepare = executor.submit(requests.post, sam_http + "/sam/prepare", json={
            "sam_model_name": "auto", "filename": "source.png", "type": "input", "subfolder": ""}, timeout=5)
        legacy = executor.submit(requests.post, sam_http + "/sam/detect", json=PROMPTS, timeout=5)
        release = executor.submit(requests.post, sam_http + "/sam/release", timeout=5)
        try:
            assert requests.get(sam_http + "/test/status?arrivals=4", timeout=2).ok
        finally:
            requests.post(sam_http + "/test/control", json={}, timeout=2).raise_for_status()
        assert_mask(detection.result())
        assert prepare.result().status_code == 200
        assert legacy.result().status_code in (200, 400)
        assert release.result().status_code == 200
    state = requests.get(sam_http + "/test/status", timeout=2).json()
    assert state["max_active"] == 1
    assert state["active"] == 0


@pytest.mark.parametrize("mode,image,prompts,status,message", [
    ("normal", b"invalid PNG", None, 400, "image"),
    ("normal", None, "{", 400, "Expecting"),
    ("normal", None, "{}", 400, "point"),
    ("missing-model", None, None, 400, "was not found"),
    ("inference-failure", None, None, 500, "Check the server log"),
])
def test_http_errors_preserve_a_successful_retry(sam_http, mode, image, prompts, status, message):
    requests.post(sam_http + "/test/control", json={"mode": mode}, timeout=2).raise_for_status()
    response = detect(sam_http, image, prompts)
    assert response.status_code == status
    assert message in response.json()["error"]
    if mode == "inference-failure":
        assert requests.get(sam_http + "/test/status", timeout=2).json()["offloaded"] >= 1
    requests.post(sam_http + "/test/control", json={}, timeout=2).raise_for_status()
    assert_mask(detect(sam_http))


def test_legacy_release_allows_preparing_the_same_image_again(sam_http):
    data = {"sam_model_name": "auto", "filename": "source.png", "type": "input", "subfolder": ""}
    assert requests.post(sam_http + "/sam/prepare", json=data, timeout=5).status_code == 200
    assert requests.post(sam_http + "/sam/release", timeout=5).status_code == 200
    assert requests.post(sam_http + "/sam/prepare", json=data, timeout=5).status_code == 200
    assert_mask(requests.post(sam_http + "/sam/detect", json=PROMPTS, timeout=5))
    assert requests.get(sam_http + "/test/status", timeout=2).json()["loaded"] == 2
