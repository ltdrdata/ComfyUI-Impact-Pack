# ComfyUI-Impact-Pack

## Custom nodes pack for ComfyUI

# Features
* MMDetLoader - Load MMDet model
* SegmDetector - Detect segmentation from input image.
* BboxDetector - Detect bbox from input image.

### ![Detector Examples](https://user-images.githubusercontent.com/128333288/228881629-0374c095-5f69-4eba-bded-773bacda33d6.png)
Detected mask example.

# Installation

1. Download 'comfyui-impact-pack.py' 
2. Copy into 'ComfyUI/custom_nodes'
3. Restart ComfyUI

# Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

dustysys/ddetailer[ddetailer](https://github.com/dustysys/ddetailer) - DDetailer

hysts/[anime-face-detector](https://github.com/hysts/anime-face-detector) - Creator of `anime-face_yolov3`, which has impressive performance on a variety of art styles.

skytnt/[anime-segmentation](https://huggingface.co/datasets/skytnt/anime-segmentation) - Synthetic dataset used to train `dd-person_mask2former`.

jerryli27/[AniSeg](https://github.com/jerryli27/AniSeg) - Annotated dataset used to train `dd-person_mask2former`.

open-mmlab/[mmdetection](https://github.com/open-mmlab/mmdetection) - Object detection toolset. `dd-person_mask2former` was trained via transfer learning using their [R-50 Mask2Former instance segmentation model](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former#instance-segmentation) as a base.

