import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import yaml
import pickle as pkl
from pathlib import Path
from src.toenet.TOENet import TOENet
from src.toenet.test import load_checkpoint


def make_discriminator_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, 5, 2, 'same'),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(64, 128, 5, 2, 'same'),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Flatten(),
        nn.Linear(1, 1)
    )


def get_discriminator_loss(cross_entropy: nn.BCELoss, denoised_images_predicted: torch.Tensor, normal_images_predicted: torch.Tensor):
    real_loss = cross_entropy(torch.ones_like(denoised_images_predicted), denoised_images_predicted)
    fake_loss = cross_entropy(torch.zeros_like(normal_images_predicted), normal_images_predicted)
    total_loss = real_loss + fake_loss
    return total_loss


def get_denoiser_loss(denoiser_criterion: nn.BCELoss, denoised_images_predicted: torch.Tensor):
    # ensure that the denoised images are classified as normal
    return denoiser_criterion(torch.ones_like(denoised_images_predicted), denoised_images_predicted)


def train(dataset: Dataset, checkpoint_dir: str) -> tuple[torch.Module, tuple[list, list]]:
    is_gpu = 1
    base_model = load_checkpoint(checkpoint_dir, is_gpu)
    discriminator_model = make_discriminator_model()
    base_model.train()
    discriminator_model.train()

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    # denoiser settings
    denoiser_optimizer = optim.Adam(base_model.parameters(), lr=config["denoiser_adam_lr"])
    denoiser_criterion = torch.nn.BCELoss()

    # discriminator settings
    discriminator_optimizer = optim.Adam(base_model.parameters(), lr=config["discriminator_adam_lr"])
    discriminator_criterion = torch.nn.BCELoss()

    num_epochs = config["num_epochs"]
    discriminator_loss_records, denoiser_loss_records = [], []
    print_loss_interval = config["print_loss_interval"]

    for epoch in range(num_epochs):
        dataloader: DataLoader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        for idx, (sand_storm_images, normal_images, _) in enumerate(dataloader):
            sand_storm_images = sand_storm_images.cuda()
            normal_images = normal_images.cuda()
            discriminator_optimizer.zero_grad()
            denoised_images = base_model(sand_storm_images)

            denoised_images_predicted_labels = discriminator_model(denoised_images)
            normal_images_predicted_labels = discriminator_model(normal_images)
            denoiser_loss = get_denoiser_loss(denoiser_criterion, denoised_images_predicted_labels)
            discriminator_loss = get_discriminator_loss(discriminator_criterion, denoised_images_predicted_labels, normal_images_predicted_labels)

            denoiser_loss.backward()
            denoiser_optimizer.step()

            discriminator_loss.backward()
            discriminator_optimizer.step()

            denoiser_loss_records.append(denoiser_loss)
            discriminator_loss_records.append(discriminator_loss)

            if idx % print_loss_interval == 0:
                print(denoiser_loss)
                print(discriminator_loss)

    # save
    torch.save(base_model.state_dict(), f"{checkpoint_dir}/base_model.pth")
    with (open(f"{checkpoint_dir}/denoiser_loss_records.pickle", "wb") as denoiser_record_file,
          open(f"{checkpoint_dir}/discriminator_loss_records.pickle", "wb") as discriminator_record_file):
        pkl.dump(denoiser_loss_records, denoiser_record_file)
        pkl.dump(discriminator_loss_records, discriminator_record_file)
    return base_model, (denoiser_loss_records, discriminator_loss_records)