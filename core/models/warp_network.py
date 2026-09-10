import numpy as np
import torch
from ..utils.load_model import load_model


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "WarpingNetwork",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, feature_3d, kp_source, kp_driving, *, return_tensor=False):
        """
        feature_3d: np.ndarray, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        """
        if self.model_type == "onnx":
            pred = self.model.run(None, {"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})
            self.model.infer()
            pred = self.model.buffer["out"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                pred = self.model(
                    torch.as_tensor(feature_3d, device=self.device),
                    torch.as_tensor(kp_source, device=self.device),
                    torch.as_tensor(kp_driving, device=self.device)
                ).float()
                if not return_tensor:
                    pred = pred.cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return pred
