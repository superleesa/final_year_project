from fire import Fire
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from train import train

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.preprocess import create_unpaired_datasets
from utils.utils import create_unique_save_dir
from pathlib import Path
import yaml

# For unpaired images training in adversarial learning
def unpaired_train_script(images_dir: str | None = None, checkpoint_dir: str | None = None, save_dir: str | None = None) -> None:

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)
    images_dir = images_dir or config["images_dir"]
    checkpoint_dir = checkpoint_dir or config["checkpoint_dir"]
    save_dir = save_dir or config["save_dir"]
    num_epochs = config["num_epochs"]

    save_dir = create_unique_save_dir(save_dir)
    datasets = create_unpaired_datasets(images_dir, num_epochs)
    
    # Train using adversarial learning approach
    _, (denoiser_loss_records, discriminator_loss_records) = train(datasets, checkpoint_dir, save_dir)
    
    # Save loss_records in csv
    unpaired_loss_df = pd.DataFrame({
        "denoiser_loss": denoiser_loss_records,
        "discriminator_loss": discriminator_loss_records,
    })
    unpaired_loss_df.to_csv(f"{save_dir}/unpaired_loss_records.csv", index=False)

    # plot for denoiser loss
    plt.plot(denoiser_loss_records)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Denoiser Loss Curve")
    plt.savefig(f"{save_dir}/unpaired_denoiser_loss_curve.png")
    plt.close()

    # plot for discriminator loss
    plt.plot(discriminator_loss_records)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Discriminator Loss Curve")
    plt.savefig(f"{save_dir}/unpaired_discriminator_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(unpaired_train_script)
