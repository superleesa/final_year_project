from torch.utils.data import DataLoader
from pathlib import Path
import sys

from train import train_loop

sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation.evaluate import evaluate
from utils.preprocess import (
    create_train_and_validation_unpaired_datasets,
    create_evaluation_dataset,
)
from utils.plots import (
    plot_and_save_metrics,
    plot_and_save_metric_distribution,
    calc_and_save_average_metrics,
    plot_and_save_loss_curve_for_multiple_runs,
    save_average_metrics_for_runs,
    plot_and_save_average_metric_barplot_for_multiple_runs,
)
from utils.utils import create_unique_save_dir, update_key_if_not_none


DENOISER_LOSS_B1_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def tune_loss_factor(
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

    base_save_dir = save_dir or params["save_dir"]

    all_runs_discriminator_loss_records = []
    all_runs_denoiser_loss_records = []
    all_runs_avg_psnr_records = []
    all_runs_avg_ssim_records = []

    for denoiser_loss_b1 in DENOISER_LOSS_B1_OPTIONS:
        params["denoiser_loss_b1"] = denoiser_loss_b1
        params["denoiser_loss_b2"] = 1 - denoiser_loss_b1

        save_dir = create_unique_save_dir(base_save_dir)
        update_key_if_not_none(params, "save_dir", base_save_dir)

        train_datasets, val_datasets = create_train_and_validation_unpaired_datasets(
            images_dir, num_epochs, train_ratio=train_ratio
        )

        update_key_if_not_none(params, "checkpoint_path", checkpoint_path)
        (
            _,
            (denoiser_loss_records, discriminator_loss_records),
            (
                val_loss_computed_indices,
                val_denoiser_loss_records,
                val_discriminator_loss_records,
            ),
        ) = train_loop(train_datasets, val_datasets, **params)
        all_runs_discriminator_loss_records.append(discriminator_loss_records)
        all_runs_denoiser_loss_records.append(denoiser_loss_records)

        # evaluation
        dataset = create_evaluation_dataset(images_dir)
        dataloader = DataLoader(dataset, batch_size=4)
        psnr_per_sample, ssim_per_sample = evaluate(
            dataloader, save_dir, checkpoint_path, save_images=save_images_type
        )
        df_metric = plot_and_save_metrics(
            {"psnr": psnr_per_sample, "ssim": ssim_per_sample}, save_dir
        )
        all_runs_avg_psnr_records.append(df_metric["psnr"].mean().item())
        all_runs_avg_ssim_records.append(df_metric["ssim"].mean().item())

    # save aggregated data
    run_names = [
        f"(b1={denoiser_loss_b1} & b2={1 - denoiser_loss_b1})"
        for denoiser_loss_b1 in DENOISER_LOSS_B1_OPTIONS
    ]
    plot_and_save_loss_curve_for_multiple_runs(
        all_runs_discriminator_loss_records, run_names, "denoiser", save_dir
    )
    plot_and_save_loss_curve_for_multiple_runs(
        all_runs_denoiser_loss_records, run_names, "discriminator", save_dir
    )
    plot_and_save_average_metric_barplot_for_multiple_runs(
        all_runs_avg_psnr_records, run_names, "PSNR", save_dir
    )
    plot_and_save_average_metric_barplot_for_multiple_runs(
        all_runs_avg_ssim_records, run_names, "SSIM", save_dir
    )
