from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import yaml
import optuna
from fire import Fire

from train import train_loop

sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation.evaluate import evaluate
from utils.preprocess import (
    create_train_and_validation_unpaired_datasets,
    create_evaluation_dataset
)
from utils.eval_plots import (
    plot_and_save_metrics,
    plot_and_save_metric_distribution,
    calc_and_save_average_metrics,
    save_aggregated_data,
)
from utils.utils import (
    create_unique_save_dir,
    update_key_if_new_value_is_not_none,
    create_dir_with_hyperparam_name,
)
from utils.train_plots import (
    save_denoiser_and_discriminator_loss_records_in_csv,
    plot_train_and_val_loss_curves,
)


def load_directory_from_yml(config_path: str | Path) -> dict:

    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)

    return {
        "checkpoint_path": config.get("checkpoint_path"),
        "images_dir": config.get("images_dir"),
        "save_dir": config.get("save_dir"),
        "eval_dir": config.get("eval_dir")
    }

def load_best_params_from_yml(best_params_path: str | Path) -> dict:

    with open(best_params_path) as ymlfile:
        best_params = yaml.safe_load(ymlfile)

    return {
        "batch_size": best_params.get("batch_size"),
        "num_epochs": best_params.get("num_epochs"),
        "denoiser_adam_lr": best_params.get("denoiser_adam_lr"),
        "discriminator_adam_lr": best_params.get("discriminator_adam_lr"),
        "denoiser_loss_b1": best_params.get("denoiser_loss_b1"),
        "denoiser_loss_b2": best_params.get("denoiser_loss_b2"),
        "print_loss_interval": best_params.get("print_loss_interval"),
        "denoiser_adversarial_loss_clip_min": best_params.get("denoiser_adversarial_loss_clip_min"),
        "denoiser_adversarial_loss_clip_max": best_params.get("denoiser_adversarial_loss_clip_max"),
        "calc_eval_loss_interval": best_params.get("calc_eval_loss_interval")
    }

def get_params(trial):
    # Generate the optimizers.
    params = {
            'denoiser_adam_lr': trial.suggest_float('denoiser_adam_lr', 1e-6, 1e-2, log = True), 
            'discriminator_adam_lr': trial.suggest_float('discriminator_adam_lr', 1e-6, 1e-2, log = True), 
            'denoiser_loss_b1' : trial.suggest_int('denoiser_loss_b1', 0.1, 0.9),
            'denoiser_loss_b2' : trial.suggest_float('denoiser_loss_b2',0.1, 0.9),
            'print_loss_interval': trial.suggest_int('print_loss_interval', 50, 500),
            'denoiser_adversarial_loss_clip_min': trial.suggest_float('denoiser_adversarial_loss_clip_min', 0.0, 1.0),
            'denoiser_adversarial_loss_clip_max': trial.suggest_float('denoiser_adversarial_loss_clip_max', 0.0, 1.0),
            'calc_eval_loss_interval': trial.suggest_int("calc_eval_loss_interval", 50, 500),
    }

    return params

def objective (trial):
    config_path = Path(__file__).parent / "config.yml"
    images_dir =  load_directory_from_yml(config_path)["images_dir"]
    num_epochs = trial.suggest_int('num_epochs', 5, 15)
    batch_size = trial.suggest_int('batch_size', 4, 24, step = 4)
    train_ratio = 0.7
    checkpoint_path = load_directory_from_yml(config_path)["checkpoint_path"]

    save_dir = load_directory_from_yml(config_path)["save_dir"]
    save_dir = create_unique_save_dir(save_dir)

    current_run_save_dir = create_dir_with_hyperparam_name(
            save_dir, get_params(trial)["denoiser_loss_b1"], get_params(trial)["denoiser_loss_b2"]
    )

    param_save_dir = current_run_save_dir

    train_datasets, val_datasets = create_train_and_validation_unpaired_datasets(
            images_dir, num_epochs, train_ratio=train_ratio
        )
        

    update_key_if_new_value_is_not_none(get_params(trial), "checkpoint_path", checkpoint_path)
    (
        _,
        (denoiser_loss_records, discriminator_loss_records),
        (
                val_loss_computed_indices,
                val_denoiser_loss_records,
                val_discriminator_loss_records,
        ),
    )= train_loop(train_datasets, val_datasets, checkpoint_path, param_save_dir, batch_size, num_epochs, **get_params(trial))

    eval_dir = load_directory_from_yml(config_path)["eval_dir"]

    # evaluation
    dataset = create_evaluation_dataset(eval_dir)
    dataloader = DataLoader(dataset, batch_size= batch_size, drop_last=True)
    psnr_per_sample, ssim_per_sample = evaluate(
        dataloader,
        save_dir,
        checkpoint_path,
        save_images= "sample"
    )

    avg_psnr = psnr_per_sample.mean().item()
    avg_ssim = ssim_per_sample.mean().item()
    
    return avg_psnr + avg_ssim

def run_tuning():
    study = optuna.create_study(direction='minimize', study_name="uft-tuning")
    study.optimize(objective, n_trials=2)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

    # Specify the full path to the file
    yaml_path = Path(__file__).parent / "best_params.yml"

    # Convert the dictionary to YAML format
    yaml_string = yaml.dump(study.best_params)

    # Write the YAML string to the specified file
    with open(yaml_path, 'w') as file:
        file.write(yaml_string)

def tuning_script():

    all_runs_discriminator_loss_records = []
    all_runs_denoiser_loss_records = []
    all_runs_avg_psnr_records = []
    all_runs_avg_ssim_records = []

    run_tuning()

    best_params_path = Path(__file__).parent / "best_params.yml"
    best_params = load_best_params_from_yml(best_params_path)

    train_ratio = 0.7
    config_path = Path(__file__).parent / "config.yml"
    images_dir =  load_directory_from_yml(config_path)["images_dir"]
    checkpoint_path = load_directory_from_yml(config_path)["checkpoint_path"]

    save_dir = load_directory_from_yml(config_path)["save_dir"]
    save_dir = create_unique_save_dir(save_dir)

    current_run_save_dir = create_dir_with_hyperparam_name(
            save_dir, best_params["denoiser_loss_b1"], best_params["denoiser_loss_b2"]
    )

    param_save_dir = current_run_save_dir
    num_epochs = best_params.pop("num_epochs")
    batch_size = best_params.pop("batch_size")


    train_datasets, val_datasets = create_train_and_validation_unpaired_datasets(
            images_dir, num_epochs, train_ratio=train_ratio
    )

    update_key_if_new_value_is_not_none(best_params, "checkpoint_path", checkpoint_path)
    (
        _,
        (denoiser_loss_records, discriminator_loss_records),
        (
                val_loss_computed_indices,
                val_denoiser_loss_records,
                val_discriminator_loss_records,
        ),
    )= train_loop(train_datasets, val_datasets, checkpoint_path, param_save_dir, batch_size, num_epochs, **best_params)

    all_runs_discriminator_loss_records.append(val_denoiser_loss_records)
    all_runs_denoiser_loss_records.append(val_discriminator_loss_records)

     # save train and val loss values for this run
    save_denoiser_and_discriminator_loss_records_in_csv(
        list(range(0, len(denoiser_loss_records))),
        denoiser_loss_records,
        discriminator_loss_records,
        param_save_dir
    )
    save_denoiser_and_discriminator_loss_records_in_csv(
        val_loss_computed_indices,
        val_denoiser_loss_records,
        val_discriminator_loss_records,
        param_save_dir
    )

    # save denoiser and discriminator loss curves
    plot_train_and_val_loss_curves(
        list(range(0, len(denoiser_loss_records))),
        val_loss_computed_indices,
        denoiser_loss_records,
        val_denoiser_loss_records,
        "Unpaired Image Training Denoiser Loss Curve",
        param_save_dir,
    )
        
    plot_train_and_val_loss_curves(
        list(range(0, len(discriminator_loss_records))),
        val_loss_computed_indices,
        discriminator_loss_records,
        val_discriminator_loss_records,
        "Unpaired Image Training Discriminator Loss Curve",
        param_save_dir,
    )

    # evaluation
    dataset = create_evaluation_dataset(images_dir)
    dataloader = DataLoader(dataset, batch_size=best_params["batch_size"])
    psnr_per_sample, ssim_per_sample = evaluate(
        dataloader,
        save_dir,
        checkpoint_path,
        save_images="sample",
    )

    # todo: there should be a function that only handles creation of metrics dataframe
    # the function below current save created plot and return df
    df_metric = plot_and_save_metrics(
        {"psnr": psnr_per_sample, "ssim": ssim_per_sample}, param_save_dir
    )

    plot_and_save_metric_distribution(df_metric["psnr"], "PSNR", param_save_dir)
    plot_and_save_metric_distribution(df_metric["ssim"], "SSIM", param_save_dir)
    calc_and_save_average_metrics(
        df_metric["psnr"], df_metric["ssim"], param_save_dir
    )

    all_runs_avg_psnr_records.append(df_metric["psnr"].mean().item())
    all_runs_avg_ssim_records.append(df_metric["ssim"].mean().item())

    # save_aggregated_data(
    #     all_runs_discriminator_loss_records,
    #     all_runs_avg_ssim_records,
    #     os.path.join(param_save_dir, "aggregated_results"),
    # )

if __name__ == "__main__":
    Fire(tuning_script)
   
        