from fire import Fire
from torch.utils.data import DataLoader
import pandas as pd
import os
from datetime import datetime

from metrics import mse_per_sample, psnr_per_sample, ssim_per_sample
from validate import validate
from utils.preprocess import create_paired_datasets


def evaluation_script(images_dir, checkpoint_dir: str, save_dir: str, save_images_type: str) -> None:
    # create unique save directory
    formatted_current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(save_dir, f'run_{formatted_current_datetime}')
    os.makedirs(save_dir, exist_ok=True)

    datasets = create_paired_datasets(images_dir)
    dataset = datasets[0]
    dataloader = DataLoader(dataset, batch_size=4)
    psnr_per_sample, ssim_per_sample = validate(dataloader, save_dir, checkpoint_dir, save_images=save_images_type)
    
    # Create DataFrame
    df_metrics = pd.DataFrame({
        'psnr': psnr_per_sample,
        'ssim': ssim_per_sample
    })
    df_metrics.to_csv(os.path.join(save_dir, 'evaluation_results.csv'), index=False)


if __name__ == "__main__":
    Fire(evaluation_script)
