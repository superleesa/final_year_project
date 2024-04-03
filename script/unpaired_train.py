from fire import Fire
import pandas as pd
import matplotlib.pyplot as plt

from train.uft import train
from utils.preprocess import create_unpaired_dataset

# For unpaired images training in adversarial learning
def unpaired_train_script(images_dir: str, checkpoint_dir: str, save_dir: str) -> None:
    save_dir = create_unique_save_dir(save_dir)
    datasets = create_unpaired_datasets(images_dir, is_train=True)
    
    # Train using adversarial learning approach
    _, uft_loss_records = train(datasets, checkpoint_dir, save_dir)
    
    # Save loss_records in csv
    unpaired_loss_df = pd.DataFrame(uft_loss_records, columns=["loss"])
    unpaired_loss_df.to_csv(f"{save_dir}/unpaired_loss_records.csv", index=False)

    # Create matplotlib plot for loss curve
    plt.plot(uft_loss_records)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Unpaired Image Training Loss Curve")
    plt.savefig(f"{save_dir}/unpaired_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    Fire(unpaired_train_script)
