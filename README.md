# ComfyUI-Impact-Pack

## Custom nodes pack for ComfyUI

# Features
* MMDetLoader - Load MMDet model
* SAMLoader - Load SAM model
* ONNXLoader - Load ONNX model
* SegmDetectorCombined - Detect segmentation and return mask from input image.
* BboxDetectorCombined - Detect bbox(bounding box) and return mask from input image.
* SamDetectorCombined - Using the technology of SAM, extract the segment at the location indicated by the input SEGS on the input image, and output it as a unified mask. 
* BitwiseAndMask - Perform 'bitwise and' operations between 2 masks
* SubtractMask - Perform subtract operations between 2 masks
* SegmDetectorForEach - Detect segmentation and return SEGS from input image.
* BboxDetectorForEach - Detect bbox(bounding box) and return SEGS from input image.
* ONNXDetectorForeach - Using the ONNX model, identify the bbox and retrieve the SEGS from the input image
* DetailerForEach - Refine image rely on SEGS.
* DetailerForEachDebug - Refine image rely on SEGS. Additionally, you can monitor cropped image and refined image of cropped image.
   * The 'DetailerForEach' and 'DetailerForEachDebug' now support an 'external_seed' that is obtained from the Seed node on the [WAS suite](https://github.com/WASasquatch/was-node-suite-comfyui)
   * To prevent the regeneration caused by the seed that does not change every time when using 'external_seed', please disable the 'seed random generate' option in the 'Detailer...' node
* BitwiseAndMaskForEach - Perform 'bitwise and' operations between 2 SEGS.
* BitwiseAndMaskForEach - Perform subtract operations between 2 SEGS.
* Segs & Masks - Perform a bitwise AND operation on SEGS and MASK.
* MaskToSegs - This node generates SEGS based on the mask. 
* ToBinaryMask - This node separates the mask generated with alpha values between 0 and 255 into 0 and 255. The non-zero parts are always set to 255.
* EmptySEGS - This node provides a empty SEGS.
* MaskPainter - This node provides a feature to draw masks.

# Installation

1. cd custom_nodes
1. git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
3. cd ComfyUI-Impact-Pack
4. (optional) python install.py
   * Impact Pack will automatically install its dependencies during its initial launch.
5. Restart ComfyUI

* You can use this colab notebook [colab notebook](https://colab.research.google.com/github/ltdrdata/ComfyUI-Impact-Pack/blob/Main/notebook/comfyui_colab_impact_pack.ipynb) to launch it. This notebook automatically downloads the impact pack to the custom_nodes directory, installs the tested dependencies, and runs it.

# Package Dependencies (If you need to manual setup.)

* pip install
   * openmim
   * segment-anything
   * pycocotools
   * onnxruntime
   
* mim install
   * mmcv==2.0.0, mmdet==3.0.0, mmengine==0.7.2
   
# Other Materials (auto-download on initial startup)

* ComfyUI/models/mmdets/bbox <= https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth
* ComfyUI/models/mmdets/bbox <= https://raw.githubusercontent.com/Bing-su/dddetailer/master/config/mmdet_anime-face_yolov3.py
* ComfyUI/models/sams <= https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Troubleshooting page
* [Troubleshooting Page](troubleshooting/TROUBLESHOOTING.md)


# How to use (DDetailer feature)

#### 1. Basic auto face detection and refine exapmle.
![example](misc/simple.png)
* The face that has been damaged due to low resolution is restored with high resolution by generating and synthesizing it, in order to restore the details.
* The FaceDetailer node is a combination of a Detector node for face detection and a Detailer node for image enhancement. See the [Advanced Tutorial](tutorial/advanced.md) for a more detailed explanation.
* Pass the MMDetLoader 's bbox model and the detection model loaded by SAMLoader to FaceDetailer . Since it performs the function of KSampler for image enhancement, it overlaps with KSampler's options.
* The MASK output of FaceDetailer provides a visualization of where the detected and enhanced areas are.

![example](misc/simple-original.png) ![example](misc/simple-refined.png)
* You can see that the face in the image on the left has increased detail as in the image on the right.

#### 2. 2Pass refine (restore a severely damaged face)
![2pass-workflow-example](misc/2pass-simple.png)
* Although two FaceDetailers can be attached together for a 2-pass configuration, various common inputs used in KSampler can be passed through DETAILER_PIPE, so FaceDetailerPipe can be used to configure easily.
* In 1pass, only rough outline recovery is required, so restore with a reasonable resolution and low options. However, if you increase the dilation at this time, not only the face but also the surrounding parts are included in the recovery range, so it is useful when you need to reshape the face other than the facial part.

![2pass-example-original](misc/2pass-original.png) ![2pass-example-middle](misc/2pass-1pass.png) ![2pass-example-result](misc/2pass-2pass.png)
* In the first stage, the severely damaged face is restored to some extent, and in the second stage, the details are restored

#### 3. Face Bbox(bounding box) + Person silhouette segmantation (prevent distortion of the background.)
![combination-workflow-example](misc/combination.png)
![combination-example-original](misc/combination-original.png) ![combination-example-refined](misc/combination-refined.png)

* Facial synthesis that emphasizes details is delicately aligned with the contours of the face, and it can be observed that it does not affect the image outside of the face.

* The BBoxDetectorForEach node is used to detect faces, and the SAMDetectorCombined node is used to find the segment related to the detected face. By using the Segs & Mask node with the two masks obtained in this way, an accurate mask that intersects based on segs can be generated. If this generated mask is input to the DetailerForEach node, only the target area can be created in high resolution from the image and then composited.

#### Mask feature

![mask-workflow-example](misc/mask.png)

* SEGS generated by the ...Detector nodes can also be converted to a MASK using nodes such as SegsCombineMask and used accordingly.

#### SAMDetection Application

![sam-workflow-example](misc/sam.png)
![sam-example-original](misc/sam-original.png) ![sam-example-masked](misc/sam-masked.png) ![sam-example-result](misc/sam-result.png) 

* By using the segmentation feature of SAM, it is possible to automatically generate the optimal mask and apply it to areas other than the face. The image on the left is the original image, the middle image is the result of applying a mask to the alpha channel, and the image on the right is the final result.

* The "detection_hint" in "SAMDetectorCombined" is a specifier that indicates which points should be included in the segmentation when performing segmentation. "center-1" specifies one point in the center of the mask, "horizontal-2" specifies two points on the center horizontal line, "vertical-2" specifies two points on the center vertical line, "rectangle-4" specifies four points in a rectangular shape inside the mask, and "diamond-4" specifies four points in a diamond shape centered around the center point.

* Unlike in face detection, for non-rigid objects, the center point may not always be the segmentation area, so be careful not to assume that the center point is always the segmentation area.

#### Mask Painter

![example](misc/maskpainter1.png)
* Click "Edit mask" button
* **Don't connect to 'mask_image' input**

![example](misc/maskpainter2.png)
* You can draw mask on Mask Painter
* Currently, this editor only provides basic functionalities

<img src="misc/maskpainter-original.png" width=352 height=512/> <img src="misc/maskpainter-result.png" width=352 height=512/>
* When used together, SAMDetector and MaskPainter can be used to enhance specific elements of an image.

# Others Tutorials
* [Advanced Tutorial](tutorial/advanced.md)
* [Mask Pointer](tutorial/maskpointer.md)
* [ONNX Tutorial](tutorial/ONNX.md)


# Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

dustysys/[ddetailer](https://github.com/dustysys/ddetailer) - DDetailer for Stable-diffusion-webUI extension.

Bing-su/[dddetailer](https://github.com/Bing-su/dddetailer) - The anime-face-detector used in ddetailer has been updated to be compatible with mmdet 3.0.0, and we have also applied a patch to the pycocotools dependency for Windows environment in ddetailer.

facebook/[segment-anything](https://github.com/facebookresearch/segment-anything) - Segmentation Anything!

hysts/[anime-face-detector](https://github.com/hysts/anime-face-detector) - Creator of `anime-face_yolov3`, which has impressive performance on a variety of art styles.

open-mmlab/[mmdetection](https://github.com/open-mmlab/mmdetection) - Object detection toolset. `dd-person_mask2former` was trained via transfer learning using their [R-50 Mask2Former instance segmentation model](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask2former#instance-segmentation) as a base.

WASasquatch/[was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui) - A powerful custom node extensions of ComfyUI.
