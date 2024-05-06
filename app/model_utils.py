import os.path
from flask import current_app
import torch
import cv2
import sys
from pathlib import Path

from toenet_prod import TOENet

sys.path.append(str(Path(__file__).parent.parent))  # add root dir
from utils.transforms import eval_transform
from utils.postprocess import postprocess


def denoise_and_save_one(model: TOENet, filename: Path) -> str:
    cv2_image = cv2.imread(str(filename))

    with torch.no_grad():
        input_image = eval_transform(cv2_image).unsqueeze(0)
        model_output = model(input_image)[0]  # get the first image of a batch
        restored_image = postprocess(model_output)[0]  # get the first image of batch again (preprocess func increases dimension)
    name, extention = os.path.splitext(filename)
    save_path = str(Path(current_app.config["upload_folder"]) / (name + "_denoised" + extention))
    cv2.imwrite(save_path, restored_image)
    return save_path

def load_model(checkpoint_path: str) -> TOENet:
    model_info = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    net = TOENet()
    device_ids = [0]
    model = torch.nn.DataParallel(net, device_ids=device_ids)
    model.load_state_dict(model_info, strict=False)
    return model
