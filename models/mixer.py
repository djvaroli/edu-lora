import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class MLPBlock(nn.Module):
    def __init__(
        self, dim: int, projection_dim: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        projection_dim = projection_dim if projection_dim else dim

        self.linear_1 = nn.Linear(dim, projection_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(projection_dim, dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.linear_1(data)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear_2(output)
        return self.dropout(output)


class MixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_patches: int,
        token_mlp_dim: int | None = None,
        channel_mlp_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.transpose = Rearrange("b h w -> b w h")
        self.token_mixing_block = MLPBlock(
            n_patches, projection_dim=token_mlp_dim, dropout=dropout
        )
        self.residual = Residual()
        self.channel_mixing_block = MLPBlock(
            dim, projection_dim=channel_mlp_dim, dropout=dropout
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MixerBlock.

        Args:
            data (torch.Tensor): tensor of shape (batch_size, n_patches, dim),
                where n_patches is the product of the number of vertical and horizontal patches.
                i.e. (batch_size, n_patches, dim) = (batch_size, n_vertical_patches * n_horizontal_patches, dim)

        Returns:
            torch.Tensor: tensor of shape (batch_size, n_patches, dim)
        """
        x = data
        y = self.layer_norm(x)
        y = self.transpose(y)
        y = self.token_mixing_block(y)
        y = self.transpose(y)
        x = self.residual(x, y)
        y = self.layer_norm(x)
        y = self.channel_mixing_block(y)
        return self.residual(x, y)


Channels = int
Height = int
Width = int


class MLPMixer(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[Channels, Height, Width],
        patch_size: int,
        hidden_dim: int,
        n_classes: int,
        n_blocks: int = 1,
        token_mlp_dim: int | None = None,
        channel_mlp_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """

        Args:
            image_size (tuple[int, int, int]): expected input image size in the format (channels, height, width).
            patch_size (int): _description_
            hidden_dim (int): _description_
            n_classes (int): _description_
            n_blocks (int, optional): _description_. Defaults to 1.
            token_mlp_dim (int | None, optional): _description_. Defaults to None.
            channel_mlp_dim (int | None, optional): _description_. Defaults to None.
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()

        channels, image_h, image_w = image_size
        assert (image_h % patch_size) == 0 and (
            image_w % patch_size
        ) == 0, f"image must be divisible by patch size. H: {image_h}, W: {image_w}, P: {patch_size}."
        n_patches = (image_h // patch_size) * (image_w // patch_size)

        # treats an image as (C, patch_size * n_vertical_patches, patch_size * n_horizontal_patches)
        # object. Each patch is C * patch_size * patch_size
        # and there are a total of n_vertical_patches * n_horizontal_patches patches.
        self.patcher = Rearrange(
            "b c (v_patches p1) (h_patches p2) -> b (v_patches h_patches) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        # (patch_size * patch_size * channels) -> hidden_dim
        p = patch_size
        self.hidden_projection = nn.Linear(p * p * channels, hidden_dim)
        self.mixer_blocks = nn.Sequential(
            *[
                MixerBlock(
                    hidden_dim, n_patches, token_mlp_dim, channel_mlp_dim, dropout
                )
                for _ in range(n_blocks)
            ]
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Reduce("b n c -> b c", "mean"),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = data
        output = self.patcher(output)
        output = self.hidden_projection(output)
        output = self.mixer_blocks(output)

        return self.classification_head(output)
