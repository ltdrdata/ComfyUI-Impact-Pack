## When a permission error occurs during the installation process (on Windows)

* There are cases where the package you are trying to install is already being used by another custom node that has been loaded.
    * This issue occurs only on Windows.
* Please close ComfyUI and execute install.py directly using Python in the custom_nodes/ComfyUI-Impact-Pack directory.
    * In case **portable** version:
        1. goto **ComfyUI_windows_portable** directory in **cmd**
        2. execute ```.\python_embeded\python -s -m custom_nodes\ComfyUI-Impact-Pack\install.py```
    * In case **venv**:
        1. activate venv
        2. execute ```python -s -m custom_nodes\ComfyUI-Impact-Pack\install.py```
    * Others:
        1. Please modify the path of 'python' according to your Python environment.
        2. execute ```(YOUR PYTHON) -s -m custom_nodes\ComfyUI-Impact-Pack\install.py```



## Destortion on Detailer

* Please also note that this issue may be caused by a bug in xformers 0.0.18. If you encounter this problem, please try adjusting the guide_size parameter.

![example](black1.png)

![example](black2.png)
* guide_size changed from 256 -> 192
