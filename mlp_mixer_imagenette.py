from argparse import ArgumentParser

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from models.mixer import MLPMixer
from training.train import train_classifier


def main(
    patch_size: int,
    hidden_dim: int,
    n_mixer_blocks: int,
    batch_size: int,
    n_epochs: int,
    lr: float = 5e-4,
    dropout: float = 0.0
):
    train_data_dict = torch.load("imagenette2-160/imagenette2-128x128-train.pt")
    val_data_dict = torch.load("imagenette2-160/imagenette2-128x128-val.pt")

    train_data = TensorDataset(train_data_dict['images'], train_data_dict['labels'].long())
    val_data = TensorDataset(val_data_dict['images'], val_data_dict['labels'].long())

    img_shape = train_data[0][0].shape

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = MLPMixer(
        image_size=img_shape,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        n_classes=10,
        n_blocks=n_mixer_blocks,
        dropout=dropout
    ).to(device)

    optimizer = Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    _ = train_classifier(
        classifier,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        device,
        n_epochs=n_epochs,
        eval_steps=0.1
    )

    classifier.eval()
    classifier.cpu()
    
    state_dict = classifier.state_dict()
    
    from pathlib import Path
    save_path = Path(wandb.run.name)
    
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    torch.save({
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "wandb_run_name": wandb.run.name,
            "model_params": {
                "patch_size": patch_size,
                "hidden_dim": hidden_dim,
                "image_size": img_shape,
                "n_classes": 10,
                "n_blocks": n_mixer_blocks,
                "dropout": dropout
            }
        }, 
        str(save_path / "mlp_mixer_imagenette.pth")
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--n_mixer_blocks", type=int)
    parser.add_argument("--dropout", type=int, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)

    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="edu-lora",

        # track hyperparameters and run metadata
        config={
            "architecture": "MLPMixer",
            "dataset": "Imagenette",
            "patch_size": args.patch_size,
            "hidden_dim": args.hidden_dim,
            "n_mixer_blocks": args.n_mixer_blocks,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.lr,
        }
    )

    main(
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        n_mixer_blocks=args.n_mixer_blocks,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        dropout=args.dropout
    )
