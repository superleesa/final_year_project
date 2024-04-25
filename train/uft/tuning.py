from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import yaml

from train import train_loop

sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation.evaluate import evaluate
from utils.preprocess import (
    create_train_and_validation_unpaired_datasets,
    create_evaluation_dataset,
)
from utils.eval_plots import (
    plot_and_save_metrics,
    plot_and_save_metric_distribution,
    calc_and_save_average_metrics,
    save_aggregated_data,
)
from utils.utils import create_unique_save_dir, update_key_if_new_value_is_not_none, create_dir_with_hyperparam_name
from utils.train_plots import save_denoiser_and_discriminator_loss_records_in_csv, plot_train_and_val_loss_curves


DENOISER_LOSS_B1_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DENOISER_LOSS_B2_OPTIONS = [1 - b1 for b1 in DENOISER_LOSS_B1_OPTIONS]

def load_uft_params_from_yml(config_path: str | Path) -> dict:

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
        "print_loss_interval": config["print_loss_interval"],
        "calc_eval_loss_interval": config["calc_eval_loss_interval"],
        "denoiser_adversarial_loss_clip_min": config.get("clip_min"),
        "denoiser_adversarial_loss_clip_max": config.get("clip_max"),
        "save_images_type": config.get("save_images_type") or "all",
    }

def tune_loss_factor(
    images_dir: str | None = None,
    checkpoint_path: str | None = None,
    save_dir: str | None = None,
) -> None:
    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    params = load_uft_params_from_yml(config_path)

    images_dir_from_config = params.pop("images_dir")
    images_dir = images_dir or images_dir_from_config
    num_epochs = params["num_epochs"]
    train_ratio = params.get("train_ratio") or 0.8

    base_save_dir = save_dir or params["save_dir"]
    base_save_dir = create_unique_save_dir(base_save_dir)

    all_runs_discriminator_loss_records = []
    all_runs_denoiser_loss_records = []
    all_runs_avg_psnr_records = []
    all_runs_avg_ssim_records = []

    for denoiser_loss_b1, denoiser_loss_b2 in zip(DENOISER_LOSS_B1_OPTIONS, DENOISER_LOSS_B2_OPTIONS):
        params["denoiser_loss_b1"] = denoiser_loss_b1
        params["denoiser_loss_b2"] = denoiser_loss_b2

        current_run_save_dir = create_dir_with_hyperparam_name(base_save_dir, denoiser_loss_b1, denoiser_loss_b2)
        params["save_dir"] = current_run_save_dir

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
        all_runs_discriminator_loss_records.append(val_denoiser_loss_records)
        all_runs_denoiser_loss_records.append(val_discriminator_loss_records)

        # save train and val loss values for this run
        save_denoiser_and_discriminator_loss_records_in_csv(list(range(0, len(denoiser_loss_records))), denoiser_loss_records, discriminator_loss_records, params['save_dir'])
        save_denoiser_and_discriminator_loss_records_in_csv(val_loss_computed_indices, val_denoiser_loss_records, val_discriminator_loss_records, params['save_dir'])

        # save denoiser and discriminator loss curves
        plot_train_and_val_loss_curves(
            list(range(0, len(denoiser_loss_records))),
            val_loss_computed_indices,
            denoiser_loss_records,
            val_denoiser_loss_records,
            "Unpaired Image Training Denoiser Loss Curve",
            params['save_dir']
        )
        plot_train_and_val_loss_curves(
            list(range(0, len(discriminator_loss_records))),
            val_loss_computed_indices,
            discriminator_loss_records,
            val_discriminator_loss_records,
            "Unpaired Image Training Discriminator Loss Curve",
            params['save_dir']
        )

        # evaluation
        dataset = create_evaluation_dataset(images_dir)
        dataloader = DataLoader(dataset, batch_size=params["batch_size"])
        psnr_per_sample, ssim_per_sample = evaluate(
            dataloader, save_dir, checkpoint_path, save_images=params["save_images_type"]
        )
        # todo: there should be a function that only handles creation of metrics dataframe
        # the function below current save created plot and return df
        df_metric = plot_and_save_metrics(
            {"psnr": psnr_per_sample, "ssim": ssim_per_sample}, params["save_dir"]
        )
        plot_and_save_metric_distribution(df_metric["psnr"], "PSNR", params["save_dir"])
        plot_and_save_metric_distribution(df_metric["ssim"], "SSIM", params["save_dir"])
        calc_and_save_average_metrics(df_metric["psnr"], df_metric["ssim"], params["save_dir"])
        all_runs_avg_psnr_records.append(df_metric["psnr"].mean().item())
        all_runs_avg_ssim_records.append(df_metric["ssim"].mean().item())

    save_aggregated_data(
        all_runs_discriminator_loss_records,
        all_runs_denoiser_loss_records,
        all_runs_avg_psnr_records,
        all_runs_avg_ssim_records,
        os.path.join(base_save_dir, "aggregated_results"),
    )
