from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import yaml
import optuna

from train import train_loop

sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation.evaluate import evaluate
from utils.preprocess import (
    create_train_and_validation_unpaired_datasets,
    create_evaluation_dataset,
)
# from utils.eval_plots import (
#     plot_and_save_metrics,
#     plot_and_save_metric_distribution,
#     calc_and_save_average_metrics,
#     save_aggregated_data,
# )
from utils.utils import (
    create_unique_save_dir,
    update_key_if_new_value_is_not_none,
    create_dir_with_hyperparam_name,
)
# from utils.train_plots import (
#     save_denoiser_and_discriminator_loss_records_in_csv,
#     plot_train_and_val_loss_curves,
# )


DENOISER_LOSS_B1_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DENOISER_LOSS_B2_OPTIONS = [1 - b1 for b1 in DENOISER_LOSS_B1_OPTIONS]


all_runs_discriminator_loss_records = []
all_runs_denoiser_loss_records = []
all_runs_avg_psnr_records = []
all_runs_avg_ssim_records = []


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
        "checkpoint_path": config.get("checkpoint_path"),
        "images_dir": config.get("images_dir"),
        "save_dir": config.get("save_dir")
    }

def get_params(trial):
    # Generate the optimizers.
    params = {
            'denoiser_adam_lr': trial.suggest_float('denoiser_adam_lr', 1e-6, 1e-2, log = True), 
            'discriminator_adam_lr': trial.suggest_float('discriminator_adam_lr', 1e-6, 1e-2, log = True), 
            'denoiser_loss_b1' : trial.suggest_int('denoiser_loss_b1', 0.1, 0.9),
            'denoiser_loss_b2' : trial.suggest_float('denoiser_loss_b2',0.1, 0.9),
            'batch_size': trial.suggest_int('batch_size', 4, 24, step = 4),
            'num_epochs':  trial.suggest_int('num_epochs', 5, 15), 
            'print_loss_interval': trial.suggest_int('print_loss_interval', 50, 500),
            'denoiser_adversarial_loss_clip_min': trial.suggest_float('denoiser_adversarial_loss_clip_min', 0.0, 1.0),
            'denoiser_adversarial_loss_clip_max': trial.suggest_float('denoiser_adversarial_loss_clip_max', 0.0, 1.0),
            'calc_eval_loss_interval': trial.suggest_int("calc_eval_loss_interval", 50, 500),
    }

    return params

def objective (trial):
    images_dir_from_config = r"/home/student/Documents/mds12/winnie/final_year_project/Data/unpaired"
    images_dir =  images_dir_from_config
    num_epochs = get_params(trial)["num_epochs"]
    train_ratio = 0.7
    checkpoint_path = r"/home/student/Documents/mds12/winnie/final_year_project/src/toenet/checkpoint/checkpoint.pth.tar"

    base_save_dir = r"/home/student/Documents/mds12/winnie/final_year_project/Data/output"
    base_save_dir = create_unique_save_dir(base_save_dir)

    current_run_save_dir = create_dir_with_hyperparam_name(
            base_save_dir, get_params(trial)["denoiser_loss_b1"], get_params(trial)["denoiser_loss_b2"]
    )

    base_save_dir = current_run_save_dir

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
    )= train_loop(train_datasets, val_datasets, checkpoint_path, current_run_save_dir, **get_params(trial))

    # Calculate the average validation loss for denoiser and discriminator
    avg_denoiser_loss = sum(val_denoiser_loss_records) / len(val_denoiser_loss_records)
    avg_discriminator_loss = sum(val_discriminator_loss_records) / len(val_discriminator_loss_records)
    
    return (avg_denoiser_loss + avg_discriminator_loss)

def run_tuning():
    study = optuna.create_study(direction='minimize', study_name="uft-tuning")
    study.optimize(objective, n_trials=10)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    run_tuning()
   
        