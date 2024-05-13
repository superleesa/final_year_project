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
    config_dir = load_directory_from_yml(config_path)

    images_dir =  config_dir["images_dir"]
    checkpoint_path = config_dir["checkpoint_path"]
    save_dir = create_unique_save_dir(config_dir["save_dir"])
    eval_dir = config_dir["eval_dir"]

    num_epochs = trial.suggest_int('num_epochs', 1, 2)
    batch_size = trial.suggest_int('batch_size', 4, 24, step = 4)
    train_ratio = 0.7


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
    
    return avg_psnr

def run_tuning():
    study = optuna.create_study(direction="maximize", study_name="uft-tuning")
    study.optimize(objective, n_trials=1)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

    # Specify the full path to the file
    yaml_path = Path(__file__).parent / "best_params.yml"

    # Convert the dictionary to YAML format
    yaml_string = yaml.dump(study.best_params)

    # Write the YAML string to the specified file
    with open(yaml_path, 'w') as file:
        file.write(yaml_string)

if __name__ == "__main__":
    Fire(run_tuning)
   
        