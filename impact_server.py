import os
from aiohttp import web
import server
import folder_paths


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
