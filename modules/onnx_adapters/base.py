from abc import ABC, abstractmethod

import numpy as np


class BaseAdapter(ABC):
    """
    Mandatory interface for all ONNX model adapters.

    This class defines the formal contract between:
    - the ONNX orchestration layer, and
    - model-specific decoding logic.

    Implementations MUST be deterministic and MUST NOT perform inference.
    """

    def __init__(self):
        # Human-readable identifier used for logging, debugging and adapter selection
        self.FAMILY = "Generic"

        # Optional: ordered list of class names exposed by the model.
        # If provided, indices MUST match model output label IDs.
        self.classes = []

    @abstractmethod
    def get_score(self, model_metadata):
        """
        Computes a compatibility score for the given model.

        This method is used during adapter selection and MUST:
        - be deterministic
        - be side-effect free
        - rely ONLY on model metadata (shapes, names, counts)

        Returns:
            float in range [0.0, 1.0]
        """
        pass

    @abstractmethod
    def preprocess(self, image_rgb, model_metadata):
        """
        Prepares the input image for model inference.

        Implementations MUST:
        - perform required color conversions (e.g. RGB → BGR)
        - add a batch dimension if required by the model
        - return a NumPy array compatible with session.run()

        Returns:
            processed_input: np.ndarray
            run_params: dict containing scaling / padding information
                        required for postprocessing
        """
        pass

    @abstractmethod
    def postprocess(self, outputs, original_shape, threshold, run_params):
        """
        Decodes raw ONNX outputs into detection results.

        Implementations MUST:
        - apply threshold filtering internally
        - map coordinates back to the original image resolution
        - return arrays with consistent shapes and dtypes

        Returns:
            labels: np.ndarray (N,) int32
            scores: np.ndarray (N,) float32
            boxes:  np.ndarray (N, 4) int32 (absolute pixel coordinates)
        """
        pass

    def get_label_name(self, label_id):
        """
        Resolves a human-readable label name from a numeric label ID.

        Falls back to a generic 'class_<id>' representation to prevent
        incorrect or misleading labels when class mappings are unavailable.
        """
        if self.classes and 0 <= label_id < len(self.classes):
            return self.classes[label_id]

        return f"class_{label_id}"

    def _get_empty_results(self):
        """
        Returns the canonical representation for 'no detections'.

        This ensures downstream nodes receive consistent, well-typed data
        and avoids special-case handling or crashes.
        """
        return (
            np.array([], dtype=np.int32),  # labels
            np.array([], dtype=np.float32),  # scores
            np.zeros((0, 4), dtype=np.int32),  # boxes
        )


class AdapterContractViolation(Exception):
    """
    Raised when an adapter violates the BaseAdapter contract.

    This includes, but is not limited to:
    - invalid return types or shapes
    - missing mandatory fields
    - semantic mismatches between adapter and model outputs
    """

    pass
