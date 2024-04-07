from fire import Fire
from torch.utils.data import DataLoader
import pandas as pd
import os
import yaml
from pathlib import Path
from evaluate import evaluate
from utils.preprocess import create_evaluation_dataset
from utils.utils import create_unique_save_dir
import matplotlib.pyplot as plt

def evaluation_script(images_dir: str | None = None, checkpoint_path: str | None = None, save_dir: str | None = None, save_images_type: str = "all") -> None:

    # load params from yml file
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path) as ymlfile:
        config = yaml.safe_load(ymlfile)
    images_dir = images_dir or config["images_dir"]
    checkpoint_path = checkpoint_path or config["checkpoint_dir"]
    save_dir = save_dir or config["save_dir"]
    save_images_type = save_images_type or config["save_images_type"]

    # create unique save directory
    save_dir = create_unique_save_dir(save_dir)

    dataset = create_evaluation_dataset(images_dir)
    # dataset = datasets[0]
    dataloader = DataLoader(dataset, batch_size=4)
    psnr_per_sample, ssim_per_sample = evaluate(dataloader, save_dir, checkpoint_path, save_images=save_images_type)
    
    # Create DataFrame
    df_metrics = pd.DataFrame({
        'psnr': psnr_per_sample,
        'ssim': ssim_per_sample
    })
    df_metrics.to_csv(os.path.join(save_dir, 'evaluation_results.csv'), index=False)

    # create histogram for psnr
    plt.hist(df_metrics["psnr"])
    plt.xlabel("PSNR")
    plt.ylabel("Number of Images")
    plt.title("Distribution of PSNR")
    plt.savefig(os.path.join(save_dir, "psnr_distribution.png"))
    plt.close()

    # create histogram for ssim
    plt.hist(df_metrics["ssim"])
    plt.xlabel("SSIM")
    plt.ylabel("Number of Images")
    plt.title("Distribution of SSIM")
    plt.savefig(os.path.join(save_dir, "ssim_distribution.png"))
    plt.close()

    # record avg psnr and ssim as yaml
    avg_psnr = df_metrics["psnr"].mean()
    avg_ssim = df_metrics["ssim"].mean()
    with open(os.path.join(save_dir, 'avg_metrics.yaml'), 'w') as f:
        yaml.dump({"avg_psnr": avg_psnr, "avg_ssim": avg_ssim}, f)


if __name__ == "__main__":
    Fire(evaluation_script)
