from fire import Fire
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import sys
sys.path.append(Path(__file__).parent.parent.parent.parent)
print(sys.path)

from utils.preprocess import create_paired_datasets
from train import train_loop
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
    train_datasets = create_paired_datasets(images_dir, num_datasets=num_epochs)
    val_datasets = create_paired_datasets(images_dir, num_datasets=num_epochs)
    
    # Train TOENet model using supervised fine-tuning
    _, train_loss_records, val_loss_records = train_loop(train_datasets, val_datasets, checkpoint_dir, save_dir)  # Checkpoints will be saved inside `save_dir`

    # Save train_loss_records in csv
    df_train_loss = pd.DataFrame(
        {"step_idx": pd.Series(range(len(train_loss_records))),
         "loss": train_loss_records})
    df_train_loss.to_csv(f"{save_dir}/train_loss_records.csv", index=False)

    # Save val_loss_records in csv
    calc_eval_loss_interval: int = config["calc_eval_loss_interval"]
    df_val_loss = pd.DataFrame({
        "step_idx": calc_eval_loss_interval * pd.Series(range(0, len(val_loss_records))),
        "loss": val_loss_records
    })
    df_val_loss.to_csv(f"{save_dir}/val_loss_records.csv", index=False)

    # Create matplotlib plot for loss curve
    plt.plot(df_train_loss["step_idx"], df_train_loss["denoiser_loss"], label="train")
    plt.plot(df_val_loss["step_idx"], df_val_loss["denoiser_loss"], label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Paired Image Training Loss Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/paired_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(paired_train_script)
    
    