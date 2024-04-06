from fire import Fire
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import sys
sys.path.append(Path(__file__).parent.parent.parent.parent)
print(sys.path)

from utils.preprocess import create_paired_datasets
from train import train
from utils.utils import create_unique_save_dir
from pathlib import Path
import yaml

# For paired images training in TOENet
def paired_train_script(images_dir: str | None = None, checkpoint_dir: str | None = None, save_dir: str | None = None) -> None:

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)
    images_dir = images_dir or config["images_dir"]
    checkpoint_dir = checkpoint_dir or config["checkpoint_dir"]
    save_dir = save_dir or config["save_dir"]
    num_epochs = config["num_epochs"]

    save_dir = create_unique_save_dir(save_dir)

    # Create paired dataset from input images directory
    datasets = create_paired_datasets(images_dir, num_datasets=num_epochs)
    
    # Train TOENet model using supervised fine-tuning
    _, sft_loss_records = train(datasets, checkpoint_dir, save_dir)  # Checkpoints will be saved inside `save_dir`

    # Save loss_records in csv
    paired_loss_df = pd.DataFrame(sft_loss_records, columns=["loss"])
    paired_loss_df.to_csv(f"{save_dir}/paired_loss_records.csv", index=False)

    # Create matplotlib plot for loss curve
    plt.plot(sft_loss_records)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Paired Image Training Loss Curve")
    plt.savefig(f"{save_dir}/paired_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(paired_train_script)
    
    