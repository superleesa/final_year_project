from utils.preprocess import create_paired_dataset, create_unpaired_dataset
from train.sft import train as train_sft
from train.uft import train as train_uft
from fire import Fire

# For paired images training in TOENet
def paired_train_script(input_image_dir, save_dir, checkpoint_dir=None):
    # Create paired dataset from input images directory
    dataset = create_paired_dataset(input_image_dir)
    
    # Train TOENet model using supervised fine-tuning
    trained_sft_model, sft_loss_records = train_sft(dataset, checkpoint_dir, save_dir)  # Checkpoints will be saved inside `save_dir`

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


def evaluation_script(y_test, trained_model):
    y_pred = trained_model(eval_data)
    print(y_pred)


if __name__ = "__main__":
    Fire({
        'paired_train': paired_train_script,
        'unpaired_train': unpaired_train_script,
        'evaluate': evaluation_script
    })
    
    