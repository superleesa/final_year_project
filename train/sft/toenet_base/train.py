import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from src.toenet.TOENet import TOENet
from train.early_stopping import SFTEarlyStopping

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.preprocess import PairedDataset
from utils.utils import load_checkpoint


def get_color_loss(
    denoised_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
    cos_sim_func: nn.CosineSimilarity,
):
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

    for batch_idx, (sand_storm_images, ground_truth_images) in tqdm(
        enumerate(val_dataloader)
    ):
        sand_storm_images, ground_truth_images = (
            sand_storm_images.cuda(),
            ground_truth_images.cuda(),
        )

        with torch.no_grad():
            batch_size = len(sand_storm_images)
            denoised_images = model(sand_storm_images)
            color_loss = get_color_loss(
                denoised_images, ground_truth_images, color_loss_criterion
            )
            l2 = l2_criterion(denoised_images, ground_truth_images)
            total_loss = loss_gamma1 * l2 + loss_gamma2 * color_loss

            loss_mean += total_loss.cpu().item() * (batch_size / (len(val_dataloader)))
    return loss_mean


def train_loop(
    train_datasets: list[PairedDataset],
    val_datasets: list[PairedDataset],
    checkpoint_path: str,
    save_dir: str,
    adam_lr: float,
    loss_gamma1: float,
    loss_gamma2: float,
    batch_size: int,
    num_epochs: int,
    print_loss_interval: int,
    calc_eval_loss_interval: int,
    early_stopping_patience: int
) -> tuple[TOENet, list[int], list[int], list[int]]:
    model = load_checkpoint(checkpoint_path, is_gpu=True)
    color_loss_criterion = nn.CosineSimilarity(dim=1)  # color channel
    l2_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=adam_lr)

    print_loss_interval = print_loss_interval or 100
    early_stopping = SFTEarlyStopping(patience=early_stopping_patience)

    loss_records = []
    val_loss_records = []
    val_loss_computed_indices = []

    for epoch_idx in tqdm(range(num_epochs), desc="epoch"):
        dataloader: DataLoader = DataLoader(
            train_datasets[epoch_idx], batch_size=batch_size, shuffle=True
        )
        for step_idx, (sand_storm_images, ground_truth_images) in tqdm(
            enumerate(dataloader), desc="step"
        ):
            model.train()

            sand_storm_images = sand_storm_images.cuda()
            ground_truth_images = ground_truth_images.cuda()

            optimizer.zero_grad()
            denoised_images = model(sand_storm_images)
            color_loss = get_color_loss(
                denoised_images, ground_truth_images, color_loss_criterion
            )
            l2 = l2_criterion(denoised_images, ground_truth_images)
            total_loss = loss_gamma1 * l2 + loss_gamma2 * color_loss
            loss_records.append(total_loss.cpu().item())
            total_loss.backward()
            optimizer.step()

            if step_idx % print_loss_interval == 0:
                print("Training Loss")
                print(f"step {epoch_idx}&{step_idx}", total_loss.item())
            

        val_dataloader = DataLoader(
            val_datasets[epoch_idx], batch_size=batch_size
        )
        val_loss = validate_loop(
            model,
            val_dataloader,
            loss_gamma1,
            loss_gamma2,
            color_loss_criterion,
            l2_criterion,
        )

        val_loss_records.append(val_loss)
        val_loss_computed_indices.append(epoch_idx)

        print("Validation Loss")
        print(f"step {epoch_idx}: ", val_loss)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # save
    torch.save(model.state_dict(), f"{save_dir}/sft_toenet_on_sie.pth")
    return model, loss_records, val_loss_records, val_loss_computed_indices
