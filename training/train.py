import logging
from typing import Tuple

import torch
from rich.logging import RichHandler
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.eval import evaluate_classifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(markup=True, rich_tracebacks=True))


def train_classifier(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 1,
    eval_steps: int | float = 0.25,
) -> tuple[nn.Module, list[float], list[float], list[float]]:
    train_steps = n_epochs * len(train_loader)

    train_loss_history = []
    eval_loss_history = []
    eval_acc_history = []

    eval_metrics = evaluate_classifier(model, eval_loader, loss_fn, device)
    eval_loss_history.append(eval_metrics["eval_loss"])
    eval_acc_history.append(eval_metrics["eval_accuracy"])

    pbar_metrics = {
        "loss": 0,
        "eval_loss": eval_metrics["eval_loss"],
        "eval_acc": eval_metrics["eval_accuracy"],
    }

    model.train()

    if eval_steps < 1:
        eval_steps = int(train_steps * eval_steps)

    elapsed_train_steps = 0
    with tqdm(total=train_steps, desc="Training model") as pbar:
        pbar.set_postfix(pbar_metrics)
        for epoch in range(n_epochs):
            for train_step, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = inputs.to(device)
                targets = targets.to(device)

                class_logits = model(inputs)
                loss = loss_fn(class_logits, targets)
                loss.backward()
                optimizer.step()

                train_loss_history.append(loss.item())
                pbar_metrics["loss"] = loss.item()

                if elapsed_train_steps > 0 and elapsed_train_steps % eval_steps == 0:
                    eval_metrics = evaluate_classifier(
                        model, eval_loader, loss_fn, device
                    )
                    eval_loss_history.append(eval_metrics["eval_loss"])
                    eval_acc_history.append(eval_metrics["eval_accuracy"])

                    pbar_metrics = {
                        "loss": loss.item(),
                        "eval_loss": eval_metrics["eval_loss"],
                        "eval_acc": eval_metrics["eval_accuracy"],
                    }
                else:
                    eval_loss_history.append(eval_loss_history[-1])
                    eval_acc_history.append(eval_acc_history[-1])

                pbar.set_postfix(pbar_metrics)
                pbar.update()
                elapsed_train_steps += 1

    return model, train_loss_history, eval_loss_history, eval_acc_history
