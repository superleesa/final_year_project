import torch

# FIXME: i think we need a copy of the TOENet model file in the app folder
from toenet_prod import TOENet


def restore_and_save_one(model: TOENet, filename: str) -> str:
    # TODO: implement
    pass

def load_model(checkpoint_dir: str) -> TOENet:
    model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar', map_location=torch.device('cpu'))
    model = TOENet()
    model.load_state_dict(model_info['state_dict'])
    return model
