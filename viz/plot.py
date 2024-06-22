import matplotlib.pyplot as plt


def plot_training_curves(
    train_loss_history: list[float],
    eval_loss_history: list[float],
    eval_acc_history: list[float],
    title: str = "Training Curves",
):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(eval_loss_history, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eval_acc_history, label="Eval Accuracy")
    plt.xlabel("Steps")
    # set the x-ticks to every 0.1
    plt.xticks(ticks=range(0, 1, 10))

    plt.ylabel("Accuracy")
    plt.legend()

    plt.suptitle(title)

    plt.show()
