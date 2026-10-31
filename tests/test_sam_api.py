"""SAM preparation HTTP contract; requires the current backend and configured model."""
import io
import os
from uuid import uuid4

import pytest
from PIL import Image
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="module")
def sam_api():
    with sync_playwright() as playwright:
        api = playwright.request.new_context(
            base_url=os.environ.get("IMPACT_E2E_BASE_URL", "http://127.0.0.1:18188"), timeout=180000)
        try:
            image = io.BytesIO()
            Image.new("RGB", (256, 256), (64, 128, 192)).save(image, format="PNG")
            response = api.post("/upload/image", multipart={
                "image": {"name": f"sam-prepare-{uuid4().hex}.png", "mimeType": "image/png", "buffer": image.getvalue()},
                "type": "input",
            })
            assert response.ok
            uploaded = response.json()
            yield api, {"sam_model_name": "auto", "filename": uploaded["name"],
                        "type": uploaded["type"], "subfolder": uploaded["subfolder"]}
        finally:
            api.dispose()


@pytest.mark.parametrize("invalid", [
    {"type": "unknown"},
    {"filename": "../outside.png"},
    {"subfolder": "../../outside"},
], ids=["unknown-type", "filename-escape", "subfolder-escape"])
def test_sam_rejects_invalid_path_on_retry(sam_api, invalid):
    api, valid = sam_api
    assert api.post("/sam/prepare", data=valid).status == 200, "Valid SAM preparation must succeed before testing rejection"
    for _ in range(2):
        response = api.post("/sam/prepare", data={**valid, **invalid})
        assert response.status == 400, "Rejected SAM preparation was cached as successful"
    assert api.post("/sam/prepare", data=valid).status == 200, "Rejected paths prevented valid preparation"
