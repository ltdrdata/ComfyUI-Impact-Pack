import sys
import subprocess


def ensure_onnx_package():
    try:
        import onnxruntime
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnxruntime'])
