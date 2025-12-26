#IMPORTS----------------------------------------------------------------------------------------------------------------
import random
from typing import Any, Dict, List

import numpy as np
import torch

#FUNCTIONS----------------------------------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    *,
    path: str,
    model: torch.nn.Module,
    model_name: str,
    class_names: List[str],
    image_size: int,
    grayscale: bool,
) -> None:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "class_names": class_names,
        "image_size": image_size,
        "grayscale": grayscale,
    }
    torch.save(payload, path)
