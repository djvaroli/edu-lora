import logging
from typing import Tuple

import wandb
import torch
from rich.logging import RichHandler
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.eval import evaluate_classifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(markup=True, rich_tracebacks=True))


TrainLossHistory = torch.Tensor
EvalLossHistory = torch.Tensor
EvalAccHistory = torch.Tensor


def train_classifier(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 1,
    eval_steps: int | float = 0.25,
) -> nn.Module:
    train_steps = n_epochs * len(train_loader)

    logger.info("Running initial model evaluation...")
    eval_metrics = evaluate_classifier(model, eval_loader, loss_fn, device)

    pbar_metrics = {
        "eval_loss": eval_metrics["eval_loss"],
        "eval_acc": eval_metrics["eval_accuracy"],
    }
    wandb.log(pbar_metrics)

    model.train()

    if eval_steps < 1:
        eval_steps = int(train_steps * eval_steps)

    logger.info("Starting training loop. Evaluating every %d steps.", eval_steps)

    elapsed_train_steps = 0
    with tqdm(total=train_steps, desc="Training model") as pbar:
        pbar.set_postfix(pbar_metrics)
        for epoch in range(n_epochs):
            pbar.set_description(f"Training model (epoch {epoch + 1}/{n_epochs})")
            for _, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = inputs.to(device)
                targets = targets.to(device)

                class_logits = model(inputs)
                loss = loss_fn(class_logits, targets)

                loss.backward()
                optimizer.step()

                pbar_metrics.update({"loss": loss.item()})

                if elapsed_train_steps > 0:
                    if elapsed_train_steps % eval_steps == 0:
                        eval_metrics = evaluate_classifier(
                            model, eval_loader, loss_fn, device
                        )

                        pbar_metrics.update({
                            "eval_loss": eval_metrics["eval_loss"],
                            "eval_acc": eval_metrics["eval_accuracy"],
                        })
                
                wandb.log(pbar_metrics)
                pbar.set_postfix(pbar_metrics)
                pbar.update()
                elapsed_train_steps += 1

    return model
