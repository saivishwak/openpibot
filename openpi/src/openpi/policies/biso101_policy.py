import dataclasses
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC, similar to LiberoInputs."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

     # If image is CHW, convert to HWC
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    return image


@dataclasses.dataclass(frozen=True)
class BiSo101Inputs(transforms.DataTransformFn):
    """
    Map a single LeRobot sample to OpenPI's internal input format.

    Expected dataset keys after repack:
      - "state": np.ndarray shape [12]
      - "image/head": third-person view RGB image (HWC or CHW)
      - optional: "image/left_wrist", "image/right_wrist"
      - "actions": np.ndarray shape [T, 12] during training
      - "prompt": string
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head = _parse_image(data["image/head"])
        left_wrist = _parse_image(data["image/left_wrist"])
        right_wrist = _parse_image(data["image/right_wrist"])

        # Build model input dict (keys MUST match what the model expects).
        inputs = {
            "state": np.asarray(data["state"]),  # shape [12]
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Training only: pass actions if present.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Language prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BiSo101Outputs(transforms.DataTransformFn):
    """
    Map model outputs back into robot action format.

    The model outputs 'actions' with shape [action_horizon, action_dim].
    Only the first 12 dims corresponding are needed for bimanual so101 joints.
    """

    def __call__(self, data: dict) -> dict:
        # Slice to 12-dim actions, then you can send that directly to the robot.
        actions = np.asarray(data["actions"])[:, :12]
        return {"actions": actions}
