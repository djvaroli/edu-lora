import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from data.mnist import load_flat_mnist_datasets
from lora import adapt_model
from training.train import train_classifier
from viz.plot import plot_training_curves


def init_mnist_classifier() -> nn.Module:
    return nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 10),
    )


# iterate over different ranks and alphas and record the final validation loss and accuracy


def main(ranks: list[int], alphas: list[float], device: str = "cpu") -> None:

    data = {"rank": [], "alpha": [], "final_val_loss": [], "final_val_acc": []}

    for r in ranks:
        for alpha in alphas:
            mnist_train, mnist_eval = load_flat_mnist_datasets()
            classifier = init_mnist_classifier()
            classifier = adapt_model(
                classifier, low_rank_adapter_type="lora", rank=r, alpha=alpha
            ).to(device)

            train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
            eval_loader = DataLoader(mnist_eval, batch_size=32, shuffle=False)

            optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

            (
                classifier,
                train_loss_history,
                eval_loss_history,
                eval_acc_history,
            ) = train_classifier(
                model=classifier,
                optimizer=optimizer,
                loss_fn=nn.CrossEntropyLoss(),
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                n_epochs=1,
            )

            data["rank"].append(r)
            data["alpha"].append(alpha)
            data["final_val_loss"].append(eval_loss_history[-1])
            data["final_val_acc"].append(eval_acc_history[-1])

    df = pd.DataFrame(data)
    df.to_csv("metrics_vs_rank.csv", index=False)


if __name__ == "__main__":
    main(ranks=[0, 1, 2, 4, 8, 16, 32], alphas=[1.0, 2.0, 4.0], device="mps")
