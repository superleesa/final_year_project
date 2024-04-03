from fire import Fire
from matplitlib.pyplot as plt
import pandas as pd

from utils.preprocess import create_paired_dataset
from train.sft import train
from utils import create_unique_save_dir

# For paired images training in TOENet
def paired_train_script(images_dir: str, checkpoint_dir: str, save_dir: str) -> None:
    save_dir = create_unique_save_dir(save_dir)

    # Create paired dataset from input images directory
    datasets = create_paired_datasets(images_dir, is_train=True)
    
    # Train TOENet model using supervised fine-tuning
    _, sft_loss_records = train(datasets, checkpoint_dir, save_dir)  # Checkpoints will be saved inside `save_dir`

    # Save loss_records in csv
    paired_loss_df = pd.DataFrame(sft_loss_records, columns=["loss"])
    paired_loss_df.to_csv(f"{save_dir}/paired_loss_records.csv", index=False)

    # Create matplotlib plot for loss curve
    plt.plot(sft_loss_records)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Paired Image Training Loss Curve")
    plt.savefig(f"{save_dir}/paired_loss_curve.png")
    plt.close()


if __name__ = "__main__":
    Fire(paired_train_script)
    
    