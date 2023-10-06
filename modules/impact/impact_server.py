import os
import threading

from aiohttp import web

import impact
import server
import folder_paths

import impact.core as core
import impact.impact_pack as impact_pack
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import nodes
from PIL import Image
import io
import impact.wildcards as wildcards
import comfy
from io import BytesIO

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

sam_lock = threading.Condition()

last_prepare_data = None


def async_prepare_sam(image_dir, model_name, filename):
    with sam_lock:
        global sam_predictor

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam_model = sam_model_registry[model_kind](checkpoint=model_name)
        sam_predictor = SamPredictor(sam_model)

        image_path = os.path.join(image_dir, filename)
        image = nodes.LoadImage().load_image(image_path)[0]
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        if impact.config.get_config()['sam_editor_cpu']:
            device = 'cpu'
        else:
            device = comfy.model_management.get_torch_device()

        sam_predictor.model.to(device=device)
        sam_predictor.set_image(image, "RGB")
        sam_predictor.model.cpu()


@server.PromptServer.instance.routes.post("/sam/prepare")
async def sam_prepare(request):
    global sam_predictor
    global last_prepare_data
    data = await request.json()

    with sam_lock:
        if last_prepare_data is not None and last_prepare_data == data:
            # already loaded: skip -- prevent redundant loading
            return web.Response(status=200)

        last_prepare_data = data

        model_name = 'sam_vit_b_01ec64.pth'
        if data['sam_model_name'] == 'auto':
            model_name = impact.config.get_config()['sam_editor_model']

        model_name = os.path.join(impact_pack.model_path, "sams", model_name)

        print(f"ComfyUI-Impact-Pack: Loading SAM model '{impact_pack.model_path}'")

        filename, image_dir = folder_paths.annotated_filepath(data["filename"])

        if image_dir is None:
            typ = data['type'] if data['type'] != '' else 'output'
            image_dir = folder_paths.get_directory_by_type(typ)
            if data['subfolder'] is not None and data['subfolder'] != '':
                image_dir += f"/{data['subfolder']}"

        if image_dir is None:
            return web.Response(status=400)

        thread = threading.Thread(target=async_prepare_sam, args=(image_dir, model_name, filename,))
        thread.start()

        print(f"ComfyUI-Impact-Pack: SAM model loaded. ")


@server.PromptServer.instance.routes.post("/sam/release")
async def release_sam(request):
    global sam_predictor

    with sam_lock:
        del sam_predictor
        sam_predictor = None

    print(f"ComfyUI-Impact-Pack: unloading SAM model")


@server.PromptServer.instance.routes.post("/sam/detect")
async def sam_detect(request):
    global sam_predictor
    with sam_lock:
        if sam_predictor is not None:
            if impact.config.get_config()['sam_editor_cpu']:
                device = 'cpu'
            else:
                device = comfy.model_management.get_torch_device()

            sam_predictor.model.to(device=device)
            try:
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
                mask = core.combine_masks2(detected_masks)

                if mask is None:
                    return web.Response(status=400)

                image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                i = 255. * image.cpu().numpy()

                img = Image.fromarray(np.clip(i[0], 0, 255).astype(np.uint8))

                img_buffer = io.BytesIO()
                img.save(img_buffer, format='png')

                headers = {'Content-Type': 'image/png'}
            finally:
                sam_predictor.model.to(device="cpu")

            return web.Response(body=img_buffer.getvalue(), headers=headers)

        else:
            return web.Response(status=400)


@server.PromptServer.instance.routes.post("/impact/wildcards")
async def populate_wildcards(request):
    data = await request.json()
    populated = wildcards.process(data['text'], data.get('seed', None))
    return web.json_response({"text": populated})


segs_picker_map = {}

@server.PromptServer.instance.routes.get("/impact/segs/picker/count")
async def segs_picker_count(request):
    node_id = request.rel_url.query.get('id', '')

    if node_id in segs_picker_map:
        res = len(segs_picker_map[node_id])
        return web.Response(status=200, text=str(res))

    return web.Response(status=400)


@server.PromptServer.instance.routes.get("/impact/segs/picker/view")
async def segs_picker(request):
    node_id = request.rel_url.query.get('id', '')
    idx = int(request.rel_url.query.get('idx', ''))

    if node_id in segs_picker_map and idx < len(segs_picker_map[node_id]):
        pil = segs_picker_map[node_id][idx]

        image_bytes = BytesIO()
        pil.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        return web.Response(status=200, body=image_bytes, content_type='image/png', headers={"Content-Disposition": f"filename={node_id}{idx}.png"})

    return web.Response(status=400)


def onprompt_for_switch(json_data):
    inversed_switch_info = {}
    onprompt_switch_info = {}

    for k, v in json_data['prompt'].items():
        if 'class_type' not in v:
            continue

        cls = v['class_type']
        if cls == 'ImpactInversedSwitch':
            select_input = v['inputs']['select']
            if isinstance(select_input, list) and len(select_input) == 2:
                input_node = json_data['prompt'][select_input[0]]
                if input_node['class_type'] == 'ImpactInt' and 'inputs' in input_node and 'value' in input_node['inputs']:
                    inversed_switch_info[k] = input_node['inputs']['value']
            else:
                inversed_switch_info[k] = select_input

        elif cls in ['ImpactSwitch', 'LatentSwitch', 'SEGSSwitch', 'ImpactMakeImageList']:
            if 'sel_mode' in v['inputs'] and v['inputs']['sel_mode']:
                select_input = v['inputs']['select']
                if isinstance(select_input, list) and len(select_input) == 2:
                    input_node = json_data['prompt'][select_input[0]]
                    if input_node['class_type'] == 'ImpactInt' and 'inputs' in input_node and 'value' in input_node['inputs']:
                        onprompt_switch_info[k] = input_node['inputs']['value']
                    if input_node['class_type'] == 'ImpactSwitch' and 'inputs' in input_node and 'select' in input_node['inputs']:
                        if isinstance(input_node['inputs']['select'], int):
                            onprompt_switch_info[k] = input_node['inputs']['select']
                        else:
                            print(f"\n##### ##### #####\n[WARN] {cls}: For the 'select' operation, only 'select_index' of the 'ImpactSwitch', which is not an input, or 'ImpactInt' and 'Primitive' are allowed as inputs.\n##### ##### #####\n")
                else:
                    onprompt_switch_info[k] = select_input

    for k, v in json_data['prompt'].items():
        disable_targets = set()

        for kk, vv in v['inputs'].items():
            if isinstance(vv, list) and len(vv) == 2:
                if vv[0] in inversed_switch_info:
                    if vv[1] + 1 != inversed_switch_info[vv[0]]:
                        disable_targets.add(kk)

        if k in onprompt_switch_info:
            selected_slot_name = f"input{onprompt_switch_info[k]}"
            for kk, vv in v['inputs'].items():
                if kk != selected_slot_name and kk.startswith('input'):
                    disable_targets.add(kk)

        for kk in disable_targets:
            del v['inputs'][kk]

    return json_data


def onprompt_for_pickers(json_data):
    detected_pickers = set()

    for k, v in json_data['prompt'].items():
        if 'class_type' not in v:
            continue

        cls = v['class_type']
        if cls == 'ImpactSEGSPicker':
            detected_pickers.add(k)

    # garbage collection
    keys_to_remove = [key for key in segs_picker_map if key not in detected_pickers]
    for key in keys_to_remove:
        del segs_picker_map[key]


def onprompt(json_data):
    json_data = onprompt_for_switch(json_data)
    onprompt_for_pickers(json_data)

    return json_data


server.PromptServer.instance.add_on_prompt_handler(onprompt)
