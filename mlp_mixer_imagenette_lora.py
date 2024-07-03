from argparse import ArgumentParser

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from rich.console import Console
from rich.table import Table

from models.mixer import MLPMixer
from models.lora import adapt_model
from training.train import train_classifier
from utils.params import count_params, count_trainable_params


def print_dict(data: dict, title: str = "Table"):
    # assume data has keys: "columns" and "data"
    console = Console()
    table = Table(title=title)
    
    for column in data["columns"]:
        table.add_column(column, justify="left", style="cyan", no_wrap=True)
    
    for row in data["data"]:
        renderable_row = []
        for item in row:
            renderable_row.append(str(item))
        
        table.add_row(*renderable_row)
    
    console.print(table)
    


def main(
    patch_size: int,
    hidden_dim: int,
    n_mixer_blocks: int,
    adapter_type: str,
    rank: int,
    alpha: float,
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
    init_num_params = count_params(classifier)
    init_num_trainable_params = count_trainable_params(classifier)
    
    wandb.run.config.update({
        "init_total_params": init_num_params,
        "init_trainable_params": init_num_trainable_params
    })
    
    classifier = adapt_model(
        classifier, 
        low_rank_adapter_type=adapter_type, 
        rank=rank, 
        alpha=alpha
    )

    lora_num_params = count_params(classifier)
    lora_num_trainable_params = count_trainable_params(classifier)
    
    wandb.run.config.update({
        "lora_total_params": lora_num_params,
        "lora_trainable_params": lora_num_trainable_params
    })
    
    optimizer = Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # print a side by side comparison of the number of parameters
    display_data = {
        "columns": ["Parameter Type", "Initial", "LoRA", "Change", "Rank"],
        "data": [
            ["Total", init_num_params, lora_num_params, round(lora_num_params / init_num_params, 2), rank],
            ["Trainable", init_num_trainable_params, lora_num_trainable_params, round(lora_num_trainable_params / init_num_trainable_params, 2), rank]
        ]
    }
    
    print_dict(display_data, title="Parameter Comparison")
    
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
                "dropout": dropout,
                "rank": rank,
                "alpha": alpha,
                "low_rank_adapter_type": "lora",
            }
        }, 
        str(save_path / "mlp_mixer_imagenette.pth")
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    # defaults are based on previous runs that yielded the highest validation accuracy
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_mixer_blocks", type=int, default=4)
    parser.add_argument("--adapter_type", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
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
            "adapter_type": args.adapter_type,
            "rank": args.rank,
            "alpha": args.alpha
        }
    )

    main(
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        n_mixer_blocks=args.n_mixer_blocks,
        adapter_type=args.adapter_type,
        rank=args.rank,
        alpha=args.alpha,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        dropout=args.dropout
    )
