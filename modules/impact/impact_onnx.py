import os
import sys
import importlib
import pkgutil

# --- Path Injection (Must be before local imports) ---
# Add current directory to sys.path to ensure adapter discovery works
# regardless of how ComfyUI loads the custom node.
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Third Party Imports ---
import numpy as np
import onnxruntime as ort

# --- Local Application Imports ---
from impact import logger, utils
import onnx_adapters
from onnx_adapters.base import AdapterContractViolation, BaseAdapter

# --- Constants ---
# Default IoU threshold used for Non-Maximum Suppression (NMS).
DEFAULT_NMS_IOU_THRESHOLD = 0.3


# ============================================================
# 1. REFINED ADAPTER DISCOVERY
# ============================================================
def get_best_adapter(model_metadata):
    """
    Scans the onnx_adapters package to find the most suitable
    implementation based on model metadata scoring.
    """
    best_adapter = None
    max_score = -1

    pkg_path = os.path.dirname(onnx_adapters.__file__)
    
    # Iterate over all modules in the adapters directory
    for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
        if module_name == "base":
            continue

        try:
            module = importlib.import_module(f"onnx_adapters.{module_name}")
        except ImportError as e:
            logger.warn(f"Failed to import adapter {module_name}: {e}")
            continue

        if hasattr(module, "Adapter"):
            adapter_class = getattr(module, "Adapter")
            
            # Instantiate to check contract
            try:
                adapter_instance = adapter_class()
            except Exception:
                continue

            # CONTRACTUAL VALIDATION: Ensure adapter inherits from BaseAdapter
            if not isinstance(adapter_instance, BaseAdapter):
                continue

            score = adapter_instance.get_score(model_metadata)
            if score > max_score:
                max_score = score
                best_adapter = adapter_instance

    return best_adapter, max_score


# ============================================================
# 2. ROBUST NMS IMPLEMENTATION (FALLBACK)
# ============================================================
def local_nms(boxes, scores, iou_threshold):
    """
    Standard CPU NMS fallback to ensure the pipeline remains
    functional if optimized libraries (torchvision) are missing or fail.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# ============================================================
# 3. PURE ORCHESTRATION PIPELINE
# ============================================================
def onnx_inference(image, onnx_model, threshold=0.3, drop_size=1):
    try:
        model_filename = os.path.basename(onnx_model)
        logger.info("-" * 50)
        logger.info(f"ORCHESTRATOR: Processing {model_filename}")

        # --- A. Session Initialization ---
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        
        # Prioritize CUDA if available, fallback to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        session = ort.InferenceSession(
            onnx_model,
            sess_options=sess_options,
            providers=providers,
        )

        inputs = session.get_inputs()
        input_meta = inputs[0]

        # --- B. Metadata Extraction & Adapter Selection ---
        model_metadata = {
            "name": model_filename.lower(),
            "input_shape": input_meta.shape,
            "output_count": len(session.get_outputs()),
            "output_names": [o.name for o in session.get_outputs()],
            "output_shapes": [o.shape for o in session.get_outputs()],
            "input_names": [i.name for i in inputs],
            "input_shapes": [i.shape for i in inputs],
            "input_name": input_meta.name,
            "input_dtype": input_meta.type,
        }

        adapter, score = get_best_adapter(model_metadata)
        
        # Adapter Confidence Checks
        if score < 0.5:
            # Critical Failure: No compatible adapter
            raise AdapterContractViolation(
                f"Low confidence ({score*100:.0f}%) for model: {model_filename}. "
                "Fix: No compatible ONNX adapter found. The model might be unsupported."
            )
        elif score < 0.8:
            # Warning: Might use a generic adapter that isn't perfect
            logger.warn(
                f"YELLOW ALERT: Selection confidence is low ({score*100:.0f}%). Accuracy may be compromised."
            )

        logger.info(f"ADAPTER: {adapter.FAMILY} (Confidence: {score*100:.0f}%)")

        # --- C. Preprocessing & Inference ---
        pil_img = utils.tensor2pil(image)
        image_rgb = np.asarray(pil_img).copy()
        orig_shape = image_rgb.shape[:2]  # [H, W]

        processed_input, run_params = adapter.preprocess(image_rgb, model_metadata)

        # Handle both Dictionary inputs (complex) and direct arrays (simple)
        if isinstance(processed_input, dict):
            raw_outputs = session.run(None, processed_input)
        else:
            raw_outputs = session.run(None, {input_meta.name: processed_input})

        # --- D. Post-processing ---
        labels, scores, boxes = adapter.postprocess(
            raw_outputs, orig_shape, threshold, run_params
        )

        # Check for adapter-reported critical errors
        if isinstance(run_params, dict) and "error" in run_params:
            raise AdapterContractViolation(run_params["error"])

        # --- E. Sanitization & Type Enforcement (Prevent 'Input type double' Error) ---
        # Instead of raising errors for wrong types, we aggressively cast them here.
        # This prevents numpy's default float64 from leaking into PyTorch.
        
        # 1. Base Type Check
        if not all(isinstance(x, np.ndarray) for x in [labels, scores, boxes]):
            # Attempt to convert list to array if possible, otherwise fail
            try:
                labels = np.array(labels)
                scores = np.array(scores)
                boxes = np.array(boxes)
            except Exception:
                raise AdapterContractViolation(
                    f"[{adapter.FAMILY}] output type error. Must be convertible to numpy arrays."
                )

        # 2. Length Consistency
        if not (len(labels) == len(scores) == len(boxes)):
            raise AdapterContractViolation(
                f"[{adapter.FAMILY}] length mismatch: L({len(labels)}) S({len(scores)}) B({len(boxes)})."
            )

        # 3. Aggressive Type Casting (Fixes RuntimeError in TAESD/Sampler)
        # Force boxes to float32 initially for math safety, will be int at end.
        boxes = boxes.astype(np.float32)
        # Force scores to float32 (Critical: prevents float64 tensors)
        scores = scores.astype(np.float32)
        # Force labels to int32
        labels = labels.astype(np.int32)

        if len(boxes) > 0:
            # 4. Finite Check
            if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
                raise AdapterContractViolation(f"[{adapter.FAMILY}] returned non-finite values (NaN/Inf).")

            # 5. Score Range
            if (scores < 0).any() or (scores > 1.001).any():
                scores = np.clip(scores, 0, 1)

            # 6. Box Shape
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise AdapterContractViolation(f"[{adapter.FAMILY}] invalid boxes shape. Expected (N, 4).")

        # --- F. Non-Maximum Suppression (NMS) ---
        if len(boxes) > 0:
            try:
                import torch
                import torchvision
                
                # Convert to Float32 Tensors (Explicit .float() is redundant if astype was used, but keeps it safe)
                t_boxes = torch.from_numpy(boxes).float()
                t_scores = torch.from_numpy(scores).float()
                
                indices = torchvision.ops.nms(
                    t_boxes, t_scores, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD
                ).numpy()
            except ImportError:
                # Fallback to local pure-numpy implementation
                logger.warn("Torchvision NMS unavailable. Using local fallback.")
                indices = local_nms(boxes, scores, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD)
            except Exception as e:
                logger.warn(f"NMS Error ({e}). Using local fallback.")
                indices = local_nms(boxes, scores, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD)

            # Filter results
            labels, scores, boxes = labels[indices], scores[indices], boxes[indices]

            # Clip boxes to image boundaries and Final Cast to Int
            boxes = np.clip(
                boxes,
                0,
                [orig_shape[1]-1, orig_shape[0]-1, orig_shape[1]-1, orig_shape[0]-1],
            ).astype(np.int32)

            logger.info(f"SUCCESS: Found {len(boxes)} valid detections after NMS.")
            
            # --- G. Label Mapping (ID -> Name) ---
            final_labels = []
            has_classes = hasattr(adapter, 'classes') and isinstance(adapter.classes, list)

            for i in range(len(boxes)):
                box = boxes[i]
                label_id = int(labels[i])
                
                if has_classes and 0 <= label_id < len(adapter.classes):
                    label_name = adapter.classes[label_id]
                else:
                    label_name = str(label_id)
                    
                final_labels.append(label_name)
                logger.info(f"  [+] {label_name}: {scores[i]:.2f} | BBox {box}")

            labels = np.array(final_labels)
        else:
            logger.warn("NOTICE: 0 detections found.")
            labels = np.array([], dtype=str) 
            scores = np.array([], dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.int32)

        # Return: labels (str array), scores (float32), boxes (int32), error_msg
        # Redundant cast on scores ensures safety even if logic above changed
        return labels, scores.astype(np.float32), boxes, None

    except Exception as e:
        logger.error(f"FATAL ORCHESTRATOR ERROR: {str(e)}")
        # Return error message for UI handling
        return None, None, None, str(e)