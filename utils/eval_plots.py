import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml


def plot_and_save_metrics(
    metric_name_to_metric_values_per_sample: dict, save_dir: str
) -> pd.DataFrame:
    """
    Create and save metric plots for psnr and ssim
    """
    # create DataFrame
    df_metrics = pd.DataFrame(metric_name_to_metric_values_per_sample)
    df_metrics.to_csv(os.path.join(save_dir, "evaluation_results.csv"), index=False)
    return df_metrics


def plot_and_save_metric_distribution(
    metric_per_image: pd.Series, metric_name: str, save_dir: str
) -> None:
    # create histogram for psnr
    plt.hist(metric_per_image)
    plt.xlabel(metric_name)
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of {metric_name}")
    plt.savefig(os.path.join(save_dir, f"{metric_name}_distribution.png"))
    plt.close()


def calc_and_save_average_metrics(
    psnr_per_sample: pd.Series, ssim_per_sample: pd.Series, save_dir: str
) -> None:
    # record avg psnr and ssim as yaml
    avg_psnr = psnr_per_sample.mean().item()
    avg_ssim = ssim_per_sample.mean().item()
    with open(os.path.join(save_dir, "avg_metrics.yaml"), "w") as f:
        yaml.dump({"avg_psnr": avg_psnr, "avg_ssim": avg_ssim}, f)


def plot_and_save_loss_curve_for_multiple_runs(
    loss_for_each_run: list[list[float]],
    test_labels: list[str],
    target_model_name: str,
    save_dir: str,
) -> None:
    # plot for psnr
    plt.figure()
    for i, psnr_per_sample in enumerate(loss_for_each_run):
        plt.plot(psnr_per_sample, label=test_labels[i])
    plt.xlabel("Image Index")
    plt.ylabel("PSNR")
    plt.title("Comparison of PSNR")
    plt.legend()
    plt.savefig(f"{save_dir}/psnr_{target_model_name}_comparison.png")
    plt.close()


def plot_and_save_average_metric_barplot_for_multiple_runs(
    avg_metrics_for_each_run: list[float],
    test_names: list[str],
    metric_name: str,
    save_dir: str,
) -> None:
    # plot for psnr
    plt.figure()
    plt.bar(test_names, avg_metrics_for_each_run)
    plt.xlabel("Test")
    plt.ylabel(f"Average {metric_name}")
    plt.title(f"Average {metric_name} for each test")
    plt.savefig(f"{save_dir}/avg_{metric_name}_comparison.png")
    plt.close()


def save_aggregated_data(
    all_runs_discriminator_loss_records: list[list[float]],
    all_runs_denoiser_loss_records: list[list[float]],
    all_runs_avg_psnr_records: list[float],
    all_runs_avg_ssim_records: list[float],
    save_dir: str,
) -> None:
    run_names = [
        f"(b1={denoiser_loss_b1} & b2={denoiser_loss_b2})"
        for denoiser_loss_b1, denoiser_loss_b2 in zip(
            DENOISER_LOSS_B1_OPTIONS, DENOISER_LOSS_B2_OPTIONS
        )
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
