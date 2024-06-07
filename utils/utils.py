import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import sys
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))
from src.toenet.TOENet import TOENet


def create_unique_save_dir(save_dir: str) -> str:
    # ensure save_dir is absolute to avoid functions creating directories in unexpected locations
    save_dir = os.path.abspath(save_dir)

    # create unique save directory
    formatted_current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(save_dir, f"run_{formatted_current_datetime}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def load_checkpoint(checkpoint_path: str, is_gpu: bool):
    if not is_gpu:
        model_info = torch.load(checkpoint_path)
        net = TOENet()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(model_info, strict=False)
    else:
        model_info = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        net = TOENet()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids)
        model.load_state_dict(model_info, strict=False)

    return model


def update_key_if_new_value_is_not_none(params: dict, key: str, new_value: Any) -> None:
    if new_value is not None:
        params[key] = new_value

def create_dir_with_hyperparam_name(base_save_dir: str, denoiser_loss_b1: float, denoiser_loss_b2: float) -> str:
    folder_name = f"b1={denoiser_loss_b1}_b2={denoiser_loss_b2}"
    path = Path(base_save_dir) / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

