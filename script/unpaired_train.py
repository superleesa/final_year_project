from utils.preprocess import create_unpaired_dataset
from train.uft import train as train_uft
from fire import Fire

# For unpaired images training in adversarial learning
def unpaired_train_script(input_image_dir, save_dir, checkpoint_dir=None):
    # Create unpaired dataset from input images directory
    dataset = create_unpaired_dataset(input_image_dir)
    
    # Train using adversarial learning approach
    trained_uft_model, uft_loss_records = train_uft(dataset, checkpoint_dir, save_dir)  # Checkpoints will be saved inside `save_dir` if not specified
    
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

if __name__ = "__main__":
    Fire(unpaired_train_script)
