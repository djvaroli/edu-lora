import torch


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the accuracy between predicted labels and target labels.

    Args:
        predictions (torch.Tensor): Predicted class logits. Shape (N, C).
        targets (torch.Tensor): True class labels. Shape (N,).
    """

    predicted_labels = predictions.argmax(dim=1)
    acc = predicted_labels.eq(targets).float().mean()
    return acc
