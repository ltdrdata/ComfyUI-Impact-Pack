import os
import shutil
import sys
import subprocess
import threading
import locale
import traceback
import re


if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version


impact_path = os.path.join(os.path.dirname(__file__), "modules")
old_subpack_path = os.path.join(os.path.dirname(__file__), "subpack")
subpack_path = os.path.join(os.path.dirname(__file__), "impact_subpack")
subpack_repo = "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


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
            

def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"[Impact Pack] EXECUTE: {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

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


pip_list = None


def get_installed_packages():
    global pip_list

    if pip_list is None:
        try:
            result = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], universal_newlines=True)
            pip_list = set([line.split()[0].lower() for line in result.split('\n') if line.strip()])
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-Manager] Failed to retrieve the information of installed pip packages.")
            return set()
    
    return pip_list
    

def is_installed(name):
    name = name.strip()
    pattern = r'([^<>!=]+)([<>!=]=?)'
    match = re.search(pattern, name)
    
    if match:
        name = match.group(1)
        
    result = name.lower() in get_installed_packages()
    return result
    

def is_requirements_installed(file_path):
    print(f"req_path: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not is_installed(line):
                    return False
                    
    return True

try:
    import platform
    import folder_paths
    from torchvision.datasets.utils import download_url
    import impact.config


    print("### ComfyUI-Impact-Pack: Check dependencies")

    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
        mim_install = [sys.executable, '-s', '-m', 'mim', 'install']
    else:
        pip_install = [sys.executable, '-m', 'pip', 'install']
        mim_install = [sys.executable, '-m', 'mim', 'install']


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

        if os.path.exists(old_subpack_path):
            shutil.rmtree(old_subpack_path)


    def remove_olds():
        global comfy_path

        comfy_path = os.path.dirname(folder_paths.__file__)
        custom_nodes_path = os.path.join(comfy_path, "custom_nodes")
        old_ini_path = os.path.join(custom_nodes_path, "impact-pack.ini")
        old_py_path = os.path.join(custom_nodes_path, "comfyui-impact-pack.py")

        if os.path.exists(impact.config.old_config_path):
            impact.config.get_config()['mmdet_skip'] = False
            os.remove(impact.config.old_config_path)

        if os.path.exists(old_ini_path):
            print(f"Delete legacy file: {old_ini_path}")
            os.remove(old_ini_path)

        if os.path.exists(old_py_path):
            print(f"Delete legacy file: {old_py_path}")
            os.remove(old_py_path)


    def ensure_pip_packages_first():
        subpack_req = os.path.join(subpack_path, "requirements.txt")
        if os.path.exists(subpack_req) and not is_requirements_installed(subpack_req):
            process_wrap(pip_install + ['-r', 'requirements.txt'], cwd=subpack_path)

        if not impact.config.get_config()['mmdet_skip']:
            process_wrap(pip_install + ['openmim'])

            try:
                import pycocotools
            except Exception:
                if platform.system() not in ["Windows"] or platform.machine() not in ["AMD64", "x86_64"]:
                    print(f"Your system is {platform.system()}; !! You need to install 'libpython3-dev' for this step. !!")

                    process_wrap(pip_install + ['pycocotools'])
                else:
                    pycocotools = {
                        (3, 8): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp38-cp38-win_amd64.whl",
                        (3, 9): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp39-cp39-win_amd64.whl",
                        (3, 10): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp310-cp310-win_amd64.whl",
                        (3, 11): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp311-cp311-win_amd64.whl",
                    }

                    version = sys.version_info[:2]
                    url = pycocotools[version]
                    process_wrap(pip_install + [url])


    def ensure_pip_packages_last():
        my_path = os.path.dirname(__file__)
        requirements_path = os.path.join(my_path, "requirements.txt")
        
        if not is_requirements_installed(requirements_path):
            process_wrap(pip_install + ['-r', requirements_path])
            
        # fallback
        try:
            import segment_anything
            from skimage.measure import label, regionprops
            import piexif
        except Exception:
            process_wrap(pip_install + ['-r', requirements_path])

        # !! cv2 importing test must be very last !!
        try:
            import cv2
        except Exception:
            try:
                if not is_installed('opencv-python'):
                    process_wrap(pip_install + ['opencv-python'])
                if not is_installed('opencv-python-headless'):
                    process_wrap(pip_install + ['opencv-python-headless'])
            except:
                print(f"[ERROR] ComfyUI-Impact-Pack: failed to install 'opencv-python'. Please, install manually.")

    def ensure_mmdet_package():
        try:
            import mmcv
            import mmdet
            from mmdet.evaluation import get_classes
        except Exception:
            process_wrap(pip_install + ['opendatalab==0.0.9'])
            process_wrap(pip_install + ['-U', 'openmim'])
            process_wrap(mim_install + ['mmcv>=2.0.0rc4, <2.1.0'])
            process_wrap(mim_install + ['mmdet==3.0.0'])
            process_wrap(mim_install + ['mmengine==0.7.4'])


    def install():
        remove_olds()

        subpack_install_script = os.path.join(subpack_path, "install.py")

        print(f"### ComfyUI-Impact-Pack: Updating subpack")
        try:
            import git
        except Exception:
            if not is_installed('GitPython'):
                process_wrap(pip_install + ['GitPython'])

        ensure_subpack()  # The installation of the subpack must take place before ensure_pip. cv2 triggers a permission error.

        if os.path.exists(subpack_install_script):
            process_wrap([sys.executable, 'install.py'], cwd=subpack_path)
            if not is_requirements_installed(os.path.join(subpack_path, 'requirements.txt')):
                process_wrap(pip_install + ['-r', 'requirements.txt'], cwd=subpack_path)
        else:
            print(f"### ComfyUI-Impact-Pack: (Install Failed) Subpack\nFile not found: `{subpack_install_script}`")

        ensure_pip_packages_first()

        if not impact.config.get_config()['mmdet_skip']:
            ensure_mmdet_package()

        ensure_pip_packages_last()

        # Download model
        print("### ComfyUI-Impact-Pack: Check basic models")

        model_path = folder_paths.models_dir

        bbox_path = os.path.join(model_path, "mmdets", "bbox")
        sam_path = os.path.join(model_path, "sams")
        onnx_path = os.path.join(model_path, "onnx")

        if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'skip_download_model')):
            if not os.path.exists(bbox_path):
                os.makedirs(bbox_path)

            if not impact.config.get_config()['mmdet_skip']:
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
