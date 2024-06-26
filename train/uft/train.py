import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import yaml
import pickle as pkl
from pathlib import Path
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm

from discriminator2 import TOENetDiscriminator as Discriminator
from trackers import SingelValueTracker, Tracker
from custom_ssim import structural_similarity_index_measure as custom_structural_similarity_index_measure

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.early_stopping import UFTEarlyStopping
from src.toenet.TOENet import TOENet
from utils.utils import load_checkpoint
from utils.preprocess import UnpairedDataset


def calc_discriminator_loss(
    cross_entropy: nn.BCELoss,
    denoised_images_predicted: torch.Tensor,
    normal_images_predicted: torch.Tensor,
) -> torch.Tensor:
    denoised_loss = cross_entropy(
        denoised_images_predicted, torch.ones_like(denoised_images_predicted)
    )
    clear_loss = cross_entropy(
        normal_images_predicted, torch.zeros_like(normal_images_predicted)
    )
    total_loss = denoised_loss + clear_loss
    return total_loss


def calc_denoiser_adversarial_loss(
    denoiser_criterion: nn.BCELoss,
    denoised_images_predicted_labels: torch.Tensor,
    clip_min: float | None = None,
    clip_max: float | None = None,
    naive_adversarial_loss_tracker: SingelValueTracker | None = None,
) -> torch.Tensor:
    # ensure that the denoised images are classified as normal
    naive_loss = denoiser_criterion(
        denoised_images_predicted_labels,
        torch.zeros_like(denoised_images_predicted_labels),
    )
    if naive_adversarial_loss_tracker is not None:
        with torch.no_grad():
            naive_adversarial_loss_tracker.add_record(naive_loss.cpu().mean().item())
    if clip_min is not None and clip_max is not None:
        return torch.clip(naive_loss, clip_min, clip_max)
    return naive_loss


def calc_denoiser_ssim_loss(
    predicted: torch.Tensor, true: torch.Tensor, use_only_structural_loss=True,
) -> torch.Tensor:
    if use_only_structural_loss:
        return 1 - custom_structural_similarity_index_measure(predicted, true)
    return 1 - structural_similarity_index_measure(predicted, true)


def calc_denoiser_loss(
    denoiser_loss_b1: float,
    denoiser_loss_b2: float,
    denoised_images: torch.Tensor,
    sand_dust_images: torch.Tensor,
    denoised_images_predicted_labels: torch.Tensor,
    denoiser_adversarial_criterion: nn.BCELoss,
    use_only_structural_loss: bool,
    clip_min: float | None = None,
    clip_max: float | None = None,
    naive_adversarial_loss_tracker: SingelValueTracker | None = None,
) -> torch.Tensor:
    return denoiser_loss_b1 * calc_denoiser_adversarial_loss(
        denoiser_adversarial_criterion,
        denoised_images_predicted_labels,
        clip_min,
        clip_max,
        naive_adversarial_loss_tracker,
    ) + denoiser_loss_b2 * calc_denoiser_ssim_loss(denoised_images, sand_dust_images, use_only_structural_loss)


def validate_loop(
    denoiser: nn.Module,
    discriminator: nn.Module,
    val_dataloader: DataLoader,
    denoiser_loss_b1: float,
    denoiser_loss_b2: float,
    denoiser_adversarial_criterion: nn.BCELoss,
    discriminator_criterion: nn.BCELoss,
    use_only_structural_loss: bool,
    clip_min: float | None,
    clip_max: float | None,
) -> tuple[float, float]:

    denoiser.eval()
    discriminator.eval()

    denoiser_loss_mean = 0.0
    discriminator_loss_mean = 0.0

    for batch_idx, (sand_dust_images, clear_images) in tqdm(enumerate(val_dataloader)):
        sand_dust_images, clear_images = sand_dust_images.cuda(), clear_images.cuda()

        with torch.no_grad():
            # denoiser loss
            batch_size = len(sand_dust_images)
            denoised_images = denoiser(sand_dust_images)
            denoised_images_predicted_labels = discriminator(denoised_images).flatten()

            denoiser_loss = calc_denoiser_loss(
                denoiser_loss_b1,
                denoiser_loss_b2,
                denoised_images,
                sand_dust_images,
                denoised_images_predicted_labels,
                denoiser_adversarial_criterion,
                use_only_structural_loss,
                clip_min,
                clip_max,
            )

            denoiser_loss_mean += denoiser_loss.cpu().item() * (
                batch_size / (len(val_dataloader))
            )

            # discriminator loss
            clear_images = clear_images.cuda()
            normal_images_predicted_labels = discriminator(clear_images).flatten()

            discriminator_loss = calc_discriminator_loss(
                discriminator_criterion,
                denoised_images_predicted_labels,
                normal_images_predicted_labels,
            )

            discriminator_loss_mean += discriminator_loss.cpu().item() * (
                batch_size / (len(val_dataloader))
            )

    return denoiser_loss_mean, discriminator_loss_mean


def train_loop(
    train_datasets: list[UnpairedDataset],
    val_datasets: list[UnpairedDataset],
    checkpoint_path: str,
    save_dir: str,
    batch_size: int,
    num_epochs: int,
    denoiser_adam_lr: float,
    discriminator_adam_lr: float,
    denoiser_loss_b1: float,
    denoiser_loss_b2: float,
    use_only_structural_loss: bool,
    denoiser_adversarial_loss_clip_min: float | None,
    denoiser_adversarial_loss_clip_max: float | None,
    print_loss_interval: int,
    calc_eval_loss_interval: int,
    early_stopping_patience: int,
    track_adv_loss: bool = False
) -> tuple[
    TOENet, tuple[list[float], list[float]], tuple[list[int], list[float], list[float]]
]:
    assert len(train_datasets) == len(val_datasets)

    is_gpu = True
    denoiser = load_checkpoint(checkpoint_path, is_gpu)  # base model already in gpu
    discriminator = Discriminator().cuda()
    early_stopping = UFTEarlyStopping(patience=early_stopping_patience)

    # denoiser settings
    denoiser_optimizer = optim.Adam(denoiser.parameters(), lr=denoiser_adam_lr)
    denoiser_adversarial_criterion = torch.nn.BCELoss()

    # discriminator settings
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=discriminator_adam_lr
    )
    discriminator_adversarial_criterion = torch.nn.BCELoss()

    denoiser_loss_records, discriminator_loss_records = [], []
    val_denoiser_loss_records, val_discriminator_loss_records = [], []
    val_loss_computed_indices = []
    global_step_counter = 0

    if track_adv_loss:
        naive_adversarial_loss_tracker = SingelValueTracker(save_dir+"/"+"adv_loss.csv", "Naive Adversarial Loss")
    else:
        naive_adversarial_loss_tracker = None

    for epoch_idx in tqdm(range(num_epochs), desc="epoch"):
        dataloader: DataLoader = DataLoader(
            train_datasets[epoch_idx], batch_size=batch_size, shuffle=True
        )

        for step_idx, (sand_dust_images, clear_images) in tqdm(
            enumerate(dataloader), desc="step"
        ):
            denoiser.train()
            discriminator.train()
            sand_dust_images = sand_dust_images.cuda()

            # update denoiser
            denoiser_optimizer.zero_grad()
            denoised_images = denoiser(sand_dust_images)

            discriminator_optimizer.zero_grad()
            denoised_images_predicted_labels = discriminator(denoised_images).flatten()

            denoiser_loss = calc_denoiser_loss(
                denoiser_loss_b1=denoiser_loss_b1,
                denoiser_loss_b2=denoiser_loss_b2,
                denoised_images=denoised_images,
                sand_dust_images=sand_dust_images,
                denoised_images_predicted_labels=denoised_images_predicted_labels,
                denoiser_adversarial_criterion=denoiser_adversarial_criterion,
                use_only_structural_loss=use_only_structural_loss,
                clip_min=denoiser_adversarial_loss_clip_min,
                clip_max=denoiser_adversarial_loss_clip_max,
                naive_adversarial_loss_tracker=naive_adversarial_loss_tracker,
            )
            denoiser_loss.backward()
            denoiser_optimizer.step()
            denoiser_loss_records.append(denoiser_loss.cpu().item())

            # update discriminator
            discriminator_optimizer.zero_grad()
            clear_images = clear_images.cuda()
            normal_images_predicted_labels = discriminator(clear_images).flatten()

            # we cannot use denoised_images_predicted_labels above because gradients are different
            # so we feed in the same set of denoised images into the discriminator again
            denoised_images_predicted_labels_for_discriminator = discriminator(
                denoised_images.detach()
            ).flatten()
            discriminator_loss = calc_discriminator_loss(
                discriminator_adversarial_criterion,
                denoised_images_predicted=denoised_images_predicted_labels_for_discriminator,
                normal_images_predicted=normal_images_predicted_labels,
            )

            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_loss_records.append(discriminator_loss.cpu().item())

            if step_idx % print_loss_interval == 0:
                print("Training Loss")
                print(
                    f"Denoiser Loss at epoch={epoch_idx}&step={step_idx}",
                    denoiser_loss_records[-1],
                )
                print(
                    f"Discriminator Loss at epoch={epoch_idx}&step={step_idx}",
                    discriminator_loss_records[-1],
                )
            
            torch.cuda.empty_cache()

        val_dataloader = DataLoader(
            val_datasets[epoch_idx], batch_size=batch_size
        )
        vaL_denoiser_loss, val_discriminator_loss = validate_loop(
            denoiser,
            discriminator,
            val_dataloader,
            denoiser_loss_b1,
            denoiser_loss_b2,
            denoiser_adversarial_criterion,
            discriminator_adversarial_criterion,
            use_only_structural_loss,
            denoiser_adversarial_loss_clip_min,
            denoiser_adversarial_loss_clip_max,
        )
        val_denoiser_loss_records.append(vaL_denoiser_loss)
        val_discriminator_loss_records.append(val_discriminator_loss)
        print("Validation Loss")
        print(
            f"Denoiser Loss at epoch={epoch_idx}:",
            vaL_denoiser_loss,
        )
        print(
            f"Discriminator Loss epoch={epoch_idx}",
            val_discriminator_loss,
        )
        val_loss_computed_indices.append(epoch_idx*step_idx)

        early_stopping(vaL_denoiser_loss, val_discriminator_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    # save
    torch.save(denoiser.state_dict(), f"{save_dir}/denoiser.pth")
    torch.save(discriminator.state_dict(), f"{save_dir}/discriminator.pth")
    if isinstance(naive_adversarial_loss_tracker, Tracker):
        naive_adversarial_loss_tracker.dump()
    
    return (
        denoiser,
        (denoiser_loss_records, discriminator_loss_records),
        (
            val_loss_computed_indices,
            val_denoiser_loss_records,
            val_discriminator_loss_records,
        ),
    )
