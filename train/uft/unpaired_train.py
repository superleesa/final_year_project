from fire import Fire
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from train import train_loop

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.preprocess import create_unpaired_datasets
from utils.utils import create_unique_save_dir
from pathlib import Path
import yaml

# For unpaired images training in adversarial learning
def unpaired_train_script(images_dir: str = None, checkpoint_dir: str = None, save_dir: str = None) -> None:

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)
    images_dir = images_dir or config["images_dir"]
    checkpoint_dir = checkpoint_dir or config["checkpoint_dir"]
    save_dir = save_dir or config["save_dir"]
    num_epochs = config["num_epochs"]

    save_dir = create_unique_save_dir(save_dir)
    train_datasets = create_unpaired_datasets(images_dir, num_epochs)
    val_datasets = create_unpaired_datasets(images_dir, num_epochs)
    
    # Train using adversarial learning approach
    _, (denoiser_loss_records, discriminator_loss_records), (val_denoiser_loss_records, val_discriminator_loss_records) = train_loop(train_datasets, val_datasets, checkpoint_dir, save_dir)
    
    # Save loss_records in csv
    df_train_unpaired_loss = pd.DataFrame({
        "step_idx": pd.Series(range(0, len(denoiser_loss_records))),
        "denoiser_loss": denoiser_loss_records,
        "discriminator_loss": discriminator_loss_records,
    })
    df_train_unpaired_loss.to_csv(f"{save_dir}/unpaired_train_loss_records.csv", index=False)

    # Save validation loss_records in csv
    calc_eval_loss_interval: int = config["calc_eval_loss_interval"]
    df_val_unpaired_loss = pd.DataFrame({
        "step_idx": calc_eval_loss_interval * pd.Series(range(0, len(val_denoiser_loss_records))),
        "denoiser_loss": val_denoiser_loss_records,
        "discriminator_loss": val_discriminator_loss_records,
    })
    df_val_unpaired_loss.to_csv(f"{save_dir}/unpaired_val_loss_records.csv", index=False)

    # plot for denoiser loss
    plt.plot(df_train_unpaired_loss["step_idx"], df_train_unpaired_loss["denoiser_loss"], label="train")
    plt.plot(df_val_unpaired_loss["step_idx"], df_val_unpaired_loss["denoiser_loss"], label="validation")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Denoiser Loss Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/unpaired_denoiser_loss_curve.png")
    plt.close()

    # plot for discriminator loss
    plt.plot(df_train_unpaired_loss["step_idx"], df_train_unpaired_loss["discriminator_loss"], label="train")
    plt.plot(df_val_unpaired_loss["step_idx"], df_val_unpaired_loss["discriminator_loss"], label="validation")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Discriminator Loss Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/unpaired_discriminator_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(unpaired_train_script)
