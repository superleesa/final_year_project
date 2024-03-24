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


def discriminator_loss(cross_entropy: nn.BCELoss, denoised_images: torch.Tensor, normal_images: torch.Tensor):
    real_loss = cross_entropy(torch.ones_like(denoised_images), denoised_images)
    fake_loss = cross_entropy(torch.zeros_like(normal_images), normal_images)
    total_loss = real_loss + fake_loss
    return total_loss


def train(dataset: Dataset, checkpoint_dir: str):
    is_gpu = 1
    base_model = load_checkpoint(checkpoint_dir, is_gpu)
    discriminator_model = make_discriminator_model()
    base_model.train()
    discriminator_model.train()

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    optimizer = optim.Adam(base_model.parameters(), lr=config["adam_lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_lr_step_size"], gamma=config["step_lr_gamma"])
    num_epochs = config["num_epochs"]
    criterion = torch.nn.BCELoss()
    loss_records = []
    print_loss_interval = config["print_loss_interval"]

    for epoch in range(num_epochs):
        dataloader: DataLoader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        for idx, (sand_storm_images, normal_images, _) in enumerate(dataloader):
            sand_storm_images = sand_storm_images.cuda()
            normal_images = normal_images.cuda()
            optimizer.zero_grad()
            denoised_images = base_model(sand_storm_images)
            predicted_class_labels_denoised = discriminator_model(denoised_images)
            predicted_class_labels_normal = discriminator_model(normal_images)
            loss = discriminator_loss(criterion, predicted_class_labels_denoised, predicted_class_labels_normal)
            loss_records.append(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % print_loss_interval == 0:
                print(loss)

    # save
    torch.save(base_model.state_dict(), f"{checkpoint_dir}/base_model.pth")
    with open(f"{checkpoint_dir}/loss_records.pickle", "wb") as f:
        pkl.dump(loss_records, f)
    return base_model, loss_records