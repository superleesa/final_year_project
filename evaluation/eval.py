from fire import Fire
from torch.utils.data import DataLoader
import pandas as pd
import os
from datetime import datetime
import yaml
from pathlib import Path
from evaluate import evaluate
from utils.preprocess import create_evaluation_dataset
from utils.utils import create_unique_save_dir

def evaluation_script(images_dir: str | None = None, checkpoint_dir: str | None = None, save_dir: str | None = None, save_images_type: str = "all") -> None:

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)
    images_dir = images_dir or config["images_dir"]
    checkpoint_dir = checkpoint_dir or config["checkpoint_dir"]
    save_dir = save_dir or config["save_dir"]
    save_images_type = save_images_type or config["save_images_type"]

    # create unique save directory
    save_dir = create_unique_save_dir(save_dir)

    datasets = create_evaluation_dataset(images_dir)
    dataset = datasets[0]
    dataloader = DataLoader(dataset, batch_size=4)
    psnr_per_sample, ssim_per_sample = evaluate(dataloader, save_dir, checkpoint_dir, save_images=save_images_type)
    
    # Create DataFrame
    df_metrics = pd.DataFrame({
        'psnr': psnr_per_sample,
        'ssim': ssim_per_sample
    })
    df_metrics.to_csv(os.path.join(save_dir, 'evaluation_results.csv'), index=False)


if __name__ == "__main__":
    Fire(evaluation_script)
