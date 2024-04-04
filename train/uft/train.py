import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import yaml
import pickle as pkl
from pathlib import Path
from src.toenet.TOENet import TOENet
from src.toenet.test import load_checkpoint
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm


def make_discriminator_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, 5, 2, 0),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(64, 128, 5, 2, 0),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Flatten(),
        nn.Linear(1095680, 1),
    )



def calc_discriminator_loss(
    cross_entropy: nn.BCELoss,
    denoised_images_predicted: torch.Tensor,
    normal_images_predicted: torch.Tensor,
) -> torch.Tensor:
    real_loss = cross_entropy(
        torch.ones_like(denoised_images_predicted), denoised_images_predicted
    )
    fake_loss = cross_entropy(
        torch.zeros_like(normal_images_predicted), normal_images_predicted
    )
    total_loss = real_loss + fake_loss
    return total_loss


def calc_denoiser_adversarial_loss(
    denoiser_criterion: nn.BCELoss,
    denoised_images_predicted_labels: torch.Tensor,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> torch.Tensor:
    # ensure that the denoised images are classified as normal
    naive_loss = denoiser_criterion(
        torch.ones_like(denoised_images_predicted_labels),
        denoised_images_predicted_labels,
    )
    if clip_min is not None and clip_max is not None:
        return torch.clip(naive_loss, clip_min, clip_max)
    return naive_loss


def calc_denoiser_ssim_loss(
    predicted: torch.Tensor, true: torch.Tensor
) -> torch.Tensor:
    return 1 - structural_similarity_index_measure(predicted, true, data_range=1.0)


def train(datasets: list[Dataset], checkpoint_dir: str, save_dir: str) -> tuple[TOENet, tuple[list[int], list[int]]]:
    is_gpu = 1
    denoiser, _, _ = load_checkpoint(checkpoint_dir, is_gpu)  # base model already in gpu
    discriminator_model = make_discriminator_model()
    denoiser.train()
    discriminator_model.train()
    discriminator_model.cuda()

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    # denoiser settings
    denoiser_optimizer = optim.Adam(
        denoiser.parameters(), lr=config["denoiser_adam_lr"]
    )
    denoiser_criterion = torch.nn.BCELoss()
    clip_min: int | float = config.get("clip_min")
    clip_max: int | float = config.get("clip_max")
    denoiser_loss_b1: float = config["denoiser_loss_b1"]
    denoiser_loss_b2: float = config["denoiser_loss_b2"]

    # discriminator settings
    discriminator_optimizer = optim.Adam(
        discriminator_model.parameters(), lr=config["discriminator_adam_lr"]
    )
    discriminator_criterion = torch.nn.BCELoss()

    num_epochs: int = config["num_epochs"]
    discriminator_loss_records, denoiser_loss_records = [], []
    print_loss_interval: int = config["print_loss_interval"]

    for epoch_idx in tqdm(range(num_epochs), desc="epoch"):
        dataloader: DataLoader = DataLoader(datasets[epoch_idx], batch_size=config["batch_size"], shuffle=True)

        for idx, (sand_dust_images, clear_images) in tqdm(enumerate(dataloader), desc="step"):
            sand_dust_images = sand_dust_images.cuda()
            
            # update denoiser
            denoiser_optimizer.zero_grad()
            denoised_images = denoiser(sand_dust_images)
            denoised_images_predicted_labels = discriminator_model(
                denoised_images
            )
            denoiser_loss = denoiser_loss_b1 * calc_denoiser_adversarial_loss(
                denoiser_criterion, denoised_images_predicted_labels, clip_min, clip_max
            ) + denoiser_loss_b2 * calc_denoiser_ssim_loss(
                denoised_images, sand_dust_images
            )
            denoiser_loss.backward()
            denoiser_optimizer.step()
            denoiser_loss_records.append(denoiser_loss.cpu().item())
            del denoiser_loss
            torch.cuda.empty_cache()


            # update discriminator
            discriminator_optimizer.zero_grad()
            clear_images = clear_images.cuda()
            normal_images_predicted_labels = discriminator_model(clear_images)
            
            # we cannot use denoised_images_predicted_labels above because gradients are different
            denoised_images_predicted_labels_for_discriminator = discriminator_model(
                denoised_images.detach()
            )
            discriminator_loss = calc_discriminator_loss(
                discriminator_criterion,
                denoised_images_predicted_labels_for_discriminator,
                normal_images_predicted_labels,
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_loss_records.append(discriminator_loss.cpu().item())

            if idx % print_loss_interval == 0:
                print(denoiser_loss_records[-1])
                print(discriminator_loss_records[-1])
            
            torch.cuda.empty_cache()

    # save
    torch.save(denoiser.state_dict(), f"{save_dir}/base_model.pth")
    return denoiser, (denoiser_loss_records, discriminator_loss_records)
