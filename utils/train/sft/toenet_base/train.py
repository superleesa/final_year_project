import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import yaml
import pickle as pkl
from torch.nn import MSELoss, CosineSimilarity
from src.tonet.TOENet import TOENet
from src.tonet.test import load_checkpoint


def get_color_loss(denoised_images: torch.Tensor, ground_truth_images: torch.Tensor, cos_sim_func: CosineSimilarity):
    batch_size, _, height, width = denoised_images.size()
    return 1 - cos_sim_func(denoised_images, ground_truth_images).mean()

def train(dataset: Dataset, checkpoint_dir: str):
    is_gpu = 1
    base_model = load_checkpoint(checkpoint_dir, is_gpu)
    base_model.train()
    color_loss_criterion = CosineSimilarity(dim=1) # color channel
    l2_criterion= MSELoss()

    # load params from yml file
    with open("config.yml") as ymlfile:
        config = yaml.safe_load(ymlfile)
    optimizer = optim.Adam(base_model.parameters(), lr=config["adam_lr"])
    loss_gamma1 = config["loss_gamma1"]
    loss_gamma2 = config["loss_gamma2"]
    num_epochs = config["num_epochs"]
    loss_records = []

    for epoch in range(num_epochs):
        dataloader: DataLoader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        for sand_storm_images, ground_truth_images in dataloader:
            optimizer.zero_grad()
            denoised_images = base_model(sand_storm_images)
            color_loss = get_color_loss(denoised_images, ground_truth_images, color_loss_criterion)
            l2 = l2_criterion(denoised_images, ground_truth_images)
            total_loss = loss_gamma1*l2 + loss_gamma2*color_loss
            loss_records.append(total_loss)
            total_loss.backward()
            optimizer.step()

    # save
    torch.save(base_model.state_dict(), f"{checkpoint_dir}/sft_toenet_on_sie.pth")
    with open(f"{checkpoint_dir}/loss_records.pickle", "wb") as f:
        pkl.dump(loss_records, f)
    return base_model, loss_records