import os
import sys
import subprocess


comfy_path = '../..'
if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version

impact_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(impact_path)
sys.path.append(comfy_path)

import platform
import folder_paths
from torchvision.datasets.utils import download_url
import impact.config


print("### ComfyUI-Impact-Pack: Check dependencies")

if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
    pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
    mim_install = [sys.executable, '-s', '-m', 'mim', 'install']
else:
    pip_install = [sys.executable, '-s', '-m', 'pip', 'install']
    mim_install = [sys.executable, '-s', '-m', 'mim', 'install']


def remove_olds():
    comfy_path = os.path.dirname(folder_paths.__file__)
    custom_nodes_path = os.path.join(comfy_path, "custom_nodes")
    old_ini_path = os.path.join(custom_nodes_path, "impact-pack.ini")
    old_py_path = os.path.join(custom_nodes_path, "comfyui-impact-pack.py")

    if os.path.exists(old_ini_path):
        print(f"Delete legacy file: {old_ini_path}")
        os.remove(old_ini_path)
    
    if os.path.exists(old_py_path):
        print(f"Delete legacy file: {old_py_path}")
        os.remove(old_py_path)


def ensure_pip_packages():
    try:
        import cv2
    except Exception:
        try:
            subprocess.check_call(pip_install + ['opencv-python'])
        except:
            print(f"ComfyUI-Impact-Pack: failed to install 'opencv-python'. Please, install manually.")

    try:
        import segment_anything
        from skimage.measure import label, regionprops
        import piexif
    except Exception:
        my_path = os.path.dirname(__file__)
        requirements_path = os.path.join(my_path, "requirements.txt")
        subprocess.check_call(pip_install + ['-r', requirements_path])

    try:
        import pycocotools
    except Exception:
        if platform.system() not in ["Windows"] or platform.machine() not in ["AMD64", "x86_64"]:
            print(f"Your system is {platform.system()}; !! You need to install 'libpython3-dev' for this step. !!")

            subprocess.check_call(pip_install + ['pycocotools'])
        else:
            pycocotools = {
                (3, 8): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp38-cp38-win_amd64.whl",
                (3, 9): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp39-cp39-win_amd64.whl",
                (3, 10): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp310-cp310-win_amd64.whl",
                (3, 11): "https://github.com/Bing-su/dddetailer/releases/download/pycocotools/pycocotools-2.0.6-cp311-cp311-win_amd64.whl",
            }

            version = sys.version_info[:2]
            url = pycocotools[version]
            subprocess.check_call(pip_install + [url])


def ensure_mmdet_package():
    try:
        import mmcv
        import mmdet
        from mmdet.evaluation import get_classes
    except Exception:
        subprocess.check_call(pip_install + ['-U', 'openmim'])
        subprocess.check_call(mim_install + ['mmcv==2.0.0'])
        subprocess.check_call(mim_install + ['mmdet==3.0.0'])
        subprocess.check_call(mim_install + ['mmengine==0.7.4'])


def install():
    remove_olds()
    ensure_pip_packages()
    ensure_mmdet_package()

    # Download model
    print("### ComfyUI-Impact-Pack: Check basic models")

    model_path = folder_paths.models_dir

    bbox_path = os.path.join(model_path, "mmdets", "bbox")
    #segm_path = os.path.join(model_path, "mmdets", "segm") -- deprecated
    sam_path = os.path.join(model_path, "sams")
    onnx_path = os.path.join(model_path, "onnx")

    if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.pth")):
        download_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)

    if not os.path.exists(os.path.join(bbox_path, "mmdet_anime-face_yolov3.py")):
        download_url("https://raw.githubusercontent.com/Bing-su/dddetailer/master/config/mmdet_anime-face_yolov3.py", bbox_path)

    if not os.path.exists(os.path.join(sam_path, "sam_vit_b_01ec64.pth")):
        download_url("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_path)

    if not os.path.exists(onnx_path):
        print(f"### ComfyUI-Impact-Pack: onnx model directory created ({onnx_path})")
        os.mkdir(onnx_path)

    impact.config.write_config(comfy_path)


install()