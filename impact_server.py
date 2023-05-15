import os
from aiohttp import web
import server
import folder_paths

import impact_core as core
import impact_pack
from segment_anything import SamPredictor
import numpy as np
import nodes

@server.PromptServer.instance.routes.post("/upload/temp")
async def upload_image(request):
    upload_dir = folder_paths.get_temp_directory()

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    post = await request.post()
    image = post.get("image")

    if image and image.file:
        filename = image.filename
        if not filename:
            return web.Response(status=400)

        split = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(upload_dir, filename)):
            filename = f"{split[0]} ({i}){split[1]}"
            i += 1

        filepath = os.path.join(upload_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image.file.read())
        
        return web.json_response({"name": filename})
    else:
        return web.Response(status=400)


sam_predictor = None
default_sam_model_name = os.path.join(impact_pack.model_path, "sams", "sam_vit_b_01ec64.pth")


@server.PromptServer.instance.routes.post("/sam/prepare")
async def load_sam_model(request):
    data = await request.json()
    
    sam_model_name = os.path.join(impact_pack.model_path, "sams", data['sam_model_name'])
    sam_predictor = SamPredictor(sam_model_name)
    
    image = nodes.LoadImage().load_image(data['img_path'])[0]
    image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    sam_predictor.set_image(image, "RGB")


@server.PromptServer.instance.routes.post("/sam/release")
async def unload_sam_model(request):
    sam_predictor = None


@server.PromptServer.instance.routes.post("/sam/detect")
async def upload_image(request):
    data = await request.json()
    
    positive_points = data['positive_points']
    negative_points = data['negative_points']
    threshold = data['threshold']

    points = []
    plabs = []

    for p in positive_points:
        points.append(p)
        plabs.append(1)

    for p in negative_points:
        points.append(p)
        plabs.append(0)
        
    detected_masks = core.sam_predict(sam_predictor, points, plabs, None, threshold)
    
    print(detected_masks)


        

    