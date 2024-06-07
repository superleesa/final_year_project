from fire import Fire
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from train import train_loop

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.preprocess import create_train_and_validation_unpaired_datasets
from utils.utils import create_unique_save_dir, update_key_if_new_value_is_not_none


def load_params_from_yml(config_path: str | Path) -> dict:

    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    return {
        "checkpoint_path": config["checkpoint_path"],
        "save_dir": config["save_dir"],
        "images_dir": config["images_dir"],
        "train_ratio": config.get("train_ratio"),
        "denoiser_adam_lr": config["denoiser_adam_lr"],
        "discriminator_adam_lr": config["discriminator_adam_lr"],
        "denoiser_loss_b1": config["denoiser_loss_b1"],
        "denoiser_loss_b2": config["denoiser_loss_b2"],
        "batch_size": config["batch_size"],
        "num_epochs": config["num_epochs"],
        "use_only_structural_loss": config["use_only_structural_loss"],
        "print_loss_interval": config["print_loss_interval"],
        "calc_eval_loss_interval": config["calc_eval_loss_interval"],
        "denoiser_adversarial_loss_clip_min": config.get("clip_min"),
        "denoiser_adversarial_loss_clip_max": config.get("clip_max"),
        "early_stopping_patience": config["early_stopping_patience"]
    }


def unpaired_train_script(
    images_dir: str | None = None,
    checkpoint_path: str | None = None,
    save_dir: str | None = None,
) -> None:

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    params = load_params_from_yml(config_path)

    images_dir_from_config = params.pop("images_dir")
    images_dir = images_dir or images_dir_from_config
    num_epochs = params["num_epochs"]
    train_ratio = params.pop("train_ratio") or 0.8

    save_dir = save_dir or params["save_dir"]
    save_dir = create_unique_save_dir(save_dir)
    update_key_if_new_value_is_not_none(params, "save_dir", save_dir)

    train_datasets, val_datasets = create_train_and_validation_unpaired_datasets(
        images_dir, num_epochs, train_ratio=train_ratio
    )

    update_key_if_new_value_is_not_none(params, "checkpoint_path", checkpoint_path)
    (
        _,
        (denoiser_loss_records, discriminator_loss_records),
        (
            val_loss_computed_indices,
            val_denoiser_loss_records,
            val_discriminator_loss_records,
        ),
    ) = train_loop(train_datasets, val_datasets, **params)

    # save loss_records in csv
    df_train_unpaired_loss = pd.DataFrame(
        {
            "step_idx": pd.Series(range(0, len(denoiser_loss_records))),
            "denoiser_loss": denoiser_loss_records,
            "discriminator_loss": discriminator_loss_records,
        }
    )
    df_train_unpaired_loss.to_csv(
        f"{save_dir}/unpaired_train_loss_records.csv", index=False
    )

    # save validation loss_records in csv
    df_val_unpaired_loss = pd.DataFrame(
        {
            "step_idx": val_loss_computed_indices,
            "denoiser_loss": val_denoiser_loss_records,
            "discriminator_loss": val_discriminator_loss_records,
        }
    )
    df_val_unpaired_loss.to_csv(
        f"{save_dir}/unpaired_val_loss_records.csv", index=False
    )

    # plot for denoiser loss
    plt.plot(
        df_train_unpaired_loss["step_idx"],
        df_train_unpaired_loss["denoiser_loss"],
        label="train",
    )
    plt.plot(
        df_val_unpaired_loss["step_idx"],
        df_val_unpaired_loss["denoiser_loss"],
        label="validation",
    )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Denoiser Loss Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/unpaired_denoiser_loss_curve.png")
    plt.close()

    # plot for discriminator loss
    plt.plot(
        df_train_unpaired_loss["step_idx"],
        df_train_unpaired_loss["discriminator_loss"],
        label="train",
    )
    plt.plot(
        df_val_unpaired_loss["step_idx"],
        df_val_unpaired_loss["discriminator_loss"],
        label="validation",
    )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Discriminator Loss Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/unpaired_discriminator_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(unpaired_train_script)
