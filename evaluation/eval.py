import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics.accuracy import compute_accuracy


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, 
    eval_loader: DataLoader, 
    loss_fn: nn.Module, 
    device: torch.device
) -> dict[str, float]:
    model.eval()

    eval_loss = torch.tensor(0.0, device=device)
    eval_accuracy = torch.tensor(0.0, device=device)
    eval_steps = 0

    for inputs, targets in eval_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        eval_loss += loss
        eval_accuracy += compute_accuracy(predictions, targets)
        eval_steps += 1

    eval_loss /= eval_steps
    eval_accuracy /= eval_steps

    model.train()

    return {"eval_loss": eval_loss.item(), "eval_accuracy": eval_accuracy.item()}
