import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable
from tqdm import tqdm

def validate_loop(model: nn.Module, val_loader: DataLoader, criterion: Callable):
    model.eval()

    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    return loss.mean().item()
