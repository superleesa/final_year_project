import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import yaml
import pickle as pkl
from torch.nn as nn
from src.toenet.TOENet import TOENet
from src.toenet.test import load_checkpoint
import os
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(Path(__file__).parent.parent.parent.parent)
from src.utils.preprocess import PairedDataset


def get_color_loss(denoised_images: torch.Tensor, ground_truth_images: torch.Tensor, cos_sim_func: nn.CosineSimilarity):
    batch_size, _, height, width = denoised_images.size()
    one = torch.tensor(1).cuda()
    return one - cos_sim_func(denoised_images, ground_truth_images).mean()

def validate_loop(
        model: nn.Module,
        val_dataloader: DataLoader,
        loss_gamma1: float,
        loss_gamma2: float,
        color_loss_criterion: nn.CosineSimilarity,
        l2_criterion: nn.MSELoss,
) -> float:

    model.eval()
    loss_mean = 0

    for batch_idx, (sand_storm_images, ground_truth_images) in tqdm(enumerate(val_dataloader)):
        sand_storm_images, ground_truth_images = sand_storm_images.cuda(), ground_truth_images.cuda()

        with torch.no_grad():
            denoised_images = model(sand_storm_images)
            color_loss = get_color_loss(denoised_images, ground_truth_images, color_loss_criterion)
            l2 = l2_criterion(denoised_images, ground_truth_images)
            total_loss = loss_gamma1 * l2 + loss_gamma2 * color_loss

            # FIXME: technically this is incorrect because the last batch might have a different size
            loss_mean += total_loss.cpu().item() * (1/(len(val_dataloader)))
    return loss_mean

def train_loop(train_datasets: list[PairedDataset], val_datasets: list[PairedDataset], checkpoint_dir: str, save_dir: str) -> tuple[TOENet, list[int], list[int]]:
    is_gpu = 1

    model, _, _ = load_checkpoint(checkpoint_dir, is_gpu)
    color_loss_criterion = nn.CosineSimilarity(dim=1) # color channel
    l2_criterion= nn.MSELoss()

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    optimizer = optim.Adam(model.parameters(), lr=config["adam_lr"])
    loss_gamma1 = config["loss_gamma1"]
    loss_gamma2 = config["loss_gamma2"]
    num_epochs = config["num_epochs"]

    print_loss_interval = config.get("print_loss_interval") or 100
    calc_eval_loss_interval = config["calc_eval_loss_interval"]

    loss_records = []
    val_loss_records = []

    for epoch_idx in tqdm(range(num_epochs), desc="epoch"):
        dataloader: DataLoader = DataLoader(train_datasets[epoch_idx], batch_size=config["batch_size"], shuffle=True)
        for step_idx, (sand_storm_images, ground_truth_images) in tqdm(enumerate(dataloader), desc="step"):
            model.train()

            sand_storm_images = sand_storm_images.cuda()
            ground_truth_images = ground_truth_images.cuda()

            optimizer.zero_grad()
            denoised_images = model(sand_storm_images)
            color_loss = get_color_loss(denoised_images, ground_truth_images, color_loss_criterion)
            l2 = l2_criterion(denoised_images, ground_truth_images)
            total_loss = loss_gamma1*l2 + loss_gamma2*color_loss
            loss_records.append(total_loss.cpu().item())
            total_loss.backward()
            optimizer.step()
            
            if step_idx % print_loss_interval == 0:
                print("Training Loss")
                print(f"step {epoch_idx}&{step_idx}", total_loss.item())

            if step_idx % calc_eval_loss_interval == 0:
                val_dataloader = DataLoader(val_datasets[epoch_idx], batch_size=config["batch_size"])
                val_loss = validate_loop(
                    model,
                    val_dataloader,
                    loss_gamma1,
                    loss_gamma2,
                    color_loss_criterion,
                    l2_criterion,
                )

                val_loss_records.append(val_loss)

                print("Validation Loss")
                print(f"step {epoch_idx}&{step_idx}", val_loss)

    # save
    torch.save(model.state_dict(), f"{save_dir}/sft_toenet_on_sie.pth")
    return model, loss_records, val_loss_records