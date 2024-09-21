import os
import shutil
import sys
import subprocess
import threading
import locale
import traceback


if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version


impact_path = os.path.join(os.path.dirname(__file__), "modules")
subpack_path = os.path.join(os.path.dirname(__file__), "impact_subpack")
subpack_repo = "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"


comfy_path = os.environ.get('COMFYUI_PATH')
if comfy_path is None:
    print(f"\n[bold yellow]WARN: The `COMFYUI_PATH` environment variable is not set. Assuming `{os.path.dirname(__file__)}/../../` as the ComfyUI path.[/bold yellow]", file=sys.stderr)
    comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

model_path = os.environ.get('COMFYUI_MODEL_PATH')
if model_path is None:
    try:
        import folder_paths
        model_path = folder_paths.models_dir
    except:
        pass

    if model_path is None:
        model_path = os.path.abspath(os.path.join(comfy_path, 'models'))
    print(f"\n[bold yellow]WARN: The `COMFYUI_MODEL_PATH` environment variable is not set. Assuming `{model_path}` as the ComfyUI path.[/bold yellow]", file=sys.stderr)


sys.path.append(impact_path)
sys.path.append(comfy_path)


# ---
def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else: 
            print(msg, end="", file=sys.stderr)
            

def process_wrap(cmd_str, cwd=None, handler=None, env=None):
    print(f"[Impact Pack] EXECUTE: {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()
# ---


try:
    import platform
    import folder_paths
    from torchvision.datasets.utils import download_url
    import impact.config

    print("### ComfyUI-Impact-Pack: Check dependencies")
    def ensure_subpack():
        import git
        if os.path.exists(subpack_path):
            try:
                repo = git.Repo(subpack_path)
                repo.remotes.origin.pull()
            except:
                traceback.print_exc()
                if platform.system() == 'Windows':
                    print(f"[ComfyUI-Impact-Pack] Please turn off ComfyUI and remove '{subpack_path}' and restart ComfyUI.")
                else:
                    shutil.rmtree(subpack_path)
                    git.Repo.clone_from(subpack_repo, subpack_path)
        else:
            git.Repo.clone_from(subpack_repo, subpack_path)


    def install():
        subpack_install_script = os.path.join(subpack_path, "install.py")

        print(f"### ComfyUI-Impact-Pack: Updating subpack")
        ensure_subpack()  # The installation of the subpack must take place before ensure_pip. cv2 triggers a permission error.

        new_env = os.environ.copy()
        new_env["COMFYUI_PATH"] = comfy_path
        new_env["COMFYUI_MODEL_PATH"] = model_path

        if os.path.exists(subpack_install_script):
            process_wrap([sys.executable, 'install.py'], cwd=subpack_path, env=new_env)
        else:
            print(f"### ComfyUI-Impact-Pack: (Install Failed) Subpack\nFile not found: `{subpack_install_script}`")

        # Download model
        print("### ComfyUI-Impact-Pack: Check basic models")
        sam_path = os.path.join(model_path, "sams")
        onnx_path = os.path.join(model_path, "onnx")

        if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'skip_download_model')):
            if not impact.config.get_config()['mmdet_skip']:
                bbox_path = os.path.join(model_path, "mmdets", "bbox")
                if not os.path.exists(bbox_path):
                    os.makedirs(bbox_path)

                if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.pth")):
                    download_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)

                if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.py")):
                    download_url("https://raw.githubusercontent.com/Bing-su/dddetailer/master/config/mmdet_anime-face_yolov3.py", bbox_path)

            if not os.path.exists(os.path.join(sam_path, "sam_vit_b_01ec64.pth")):
                download_url("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_path)

        if not os.path.exists(onnx_path):
            print(f"### ComfyUI-Impact-Pack: onnx model directory created ({onnx_path})")
            os.mkdir(onnx_path)

        impact.config.write_config()

    install()

except Exception as e:
    print("[ERROR] ComfyUI-Impact-Pack: Dependency installation has failed. Please install manually.")
    traceback.print_exc()
