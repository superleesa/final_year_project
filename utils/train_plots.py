import pandas as pd
import matplotlib.pyplot as plt


def save_loss_records_in_csv(
    step_idx: pd.Series[int],
    loss_records: pd.Series[float],
    loss_label: str,
    save_dir: str,
) -> None:
    df_train_loss = pd.DataFrame(
        {
            "step_idx": step_idx,
            f"{loss_label}_loss": loss_records,
        }
    )
    df_train_loss.to_csv(f"{save_dir}/{loss_label}_loss_records.csv", index=False)


def plot_train_and_val_loss_curves(
    train_step_idx: int,
    val_step_idx: int,
    train_loss_records: list[float],
    val_loss_records: list[float],
    plot_title: str,
    save_dir: str,
) -> None:
    plt.plot(
        train_step_idx,
        train_loss_records,
        label="train",
    )
    plt.plot(
        val_step_idx,
        val_loss_records,
        label="validation",
    )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(plot_title)
    plt.legend()
    plt.savefig(f"{save_dir}/{plot_title.lower().replace(' ', '_')}.png")
    plt.close()
