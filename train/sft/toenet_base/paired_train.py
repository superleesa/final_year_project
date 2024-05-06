from fire import Fire
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
print(sys.path)

from utils.preprocess import create_train_and_validation_paired_datasets
from train import train_loop
from utils.utils import create_unique_save_dir, update_key_if_not_none


def load_params_from_yml(config_path: str | Path) -> dict:

    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    return {
        "checkpoint_path": config["checkpoint_path"],
        "save_dir": config["save_dir"],
        "images_dir": config["images_dir"],
        "train_ratio": config.get("train_ratio"),
        "adam_lr": config["adam_lr"],
        "loss_gamma1": config["loss_gamma1"],
        "loss_gamma2": config["loss_gamma2"],
        "batch_size": config["batch_size"],
        "num_epochs": config["num_epochs"],
        "print_loss_interval": config["print_loss_interval"],
        "calc_eval_loss_interval": config["calc_eval_loss_interval"],
    }


def paired_train_script(
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
    train_ratio = params.get("train_ratio") or 0.8

    save_dir = save_dir or params["save_dir"]
    save_dir = create_unique_save_dir(save_dir)
    update_key_if_not_none(params, "save_dir", save_dir)

    train_datasets, val_datasets = create_train_and_validation_paired_datasets(
        images_dir, num_datasets=num_epochs, train_ratio=train_ratio
    )

    update_key_if_not_none(params, "checkpoint_path", checkpoint_path)
    _, train_loss_records, val_loss_records, val_loss_computed_indices = train_loop(
        train_datasets, val_datasets, **params
    )  # Checkpoints will be saved inside `save_dir`

    # Save train_loss_records in csv
    df_train_loss = pd.DataFrame(
        {
            "step_idx": pd.Series(range(len(train_loss_records))),
            "loss": train_loss_records,
        }
    )
    df_train_loss.to_csv(f"{save_dir}/train_loss_records.csv", index=False)

    # save val_loss_records in csv
    calc_eval_loss_interval: int = params["calc_eval_loss_interval"]
    df_val_loss = pd.DataFrame(
        {"step_idx": val_loss_computed_indices, "loss": val_loss_records}
    )
    df_val_loss.to_csv(f"{save_dir}/val_loss_records.csv", index=False)

    # create matplotlib plot for loss curve
    plt.plot(df_train_loss["step_idx"], df_train_loss["loss"], label="train")
    plt.plot(df_val_loss["step_idx"], df_val_loss["loss"], label="validation")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Paired Image Training Loss Curve")
    plt.legend()
    plt.savefig(f"{save_dir}/paired_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(paired_train_script)
