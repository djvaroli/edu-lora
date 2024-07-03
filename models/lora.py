from typing import Callable, Literal

import torch
from torch import nn
from torch import functional as F


class LowRankAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        b_init: Callable[[int, int], torch.Tensor] = torch.zeros,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0"

        # scaling for weights
        std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn((in_features, rank)) * std)

        # init B to zeros s.t. initially outputs of LoRA module match of original module
        self.B = nn.Parameter(b_init(rank, out_features))

        if device is not None:
            self.to(device)
        
    def _reinit_B(
        self, b_init: Callable[[int, int], torch.Tensor] = torch.zeros
    ) -> None:
        self.B.data = b_init(*self.B.shape)

    def get_expanded_weights(self) -> nn.Parameter:
        return self.A @ self.B

    @property
    def rank(self) -> int:
        return self.A.shape[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.A @ self.B


class LowRankLinearAdapterBase(nn.Module):
    @classmethod
    def from_linear(
        cls, linear: nn.Linear, rank: int, alpha: float = 1.0
    ) -> "LowRankLinearAdapterBase":
        raise NotImplementedError

    def merge_weights(self) -> None:
        raise NotImplementedError

    def unmerge_weights(self) -> None:
        raise NotImplementedError


class LowRankLinearAdapter(LowRankLinearAdapterBase):
    """ """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()

        # track the original weights
        self._linear = linear
        self._alpha = alpha
        self._rank = rank
        self._lora = None
        self._merged = False

        if rank > 0:
            self._linear.weight.requires_grad = False

            # create a low-rank adapter
            out_dim, in_dim = self._linear.weight.shape
            self._lora = LowRankAdapter(in_dim, out_dim, rank, device=self._linear.weight.device)

    def merge_weights(self) -> None:
        if self._rank > 0:
            lora_weights_expanded = self._lora.get_expanded_weights()
            self._linear.weight.data += self._alpha * lora_weights_expanded.T
            self._merged = True

    def unmerge_weights(self) -> None:
        if self._rank > 0:
            lora_weights_expanded = self._lora.get_expanded_weights()
            self._linear.weight.data -= self._alpha * lora_weights_expanded.T
            self._merged = False

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, rank: int, alpha: float = 1.0
    ) -> "LowRankLinearAdapter":
        return cls(linear, rank, alpha)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # forward pass through original weights
        
        if self._linear.bias is not None:
            linear_out = nn.functional.linear(inputs, self._linear.weight, self._linear.bias)
        else:
            linear_out = nn.functional.linear(inputs, self._linear.weight)

        if self._merged or self._lora is None:
            # if the weights have been merged, LoRA updates included in self._linear.weight
            # if no low-rank adapter, return the original output
            return linear_out

        return linear_out + self._alpha * self._lora(inputs)

    def extra_repr(self) -> str:
        return f'r={self._rank}, alpha={self._alpha}, merged={self._merged}'

class WeightDecomposedLinearAdapater(LowRankLinearAdapterBase):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()

        # track the original weights
        self._linear = linear
        self._alpha = alpha
        self._rank = rank
        self._lora = None

        if rank > 0:
            self._linear.weight.requires_grad = False

            out_dim, in_dim = self._linear.weight.shape

            # vector of magnitudes, initialized as the ||W_0||_c (taken along the columns)
            # recall that weights are stored as (out_features, in_features) matrices,
            # so m has shape (1, in_features)
            self.m = nn.Parameter(self._linear.weight.norm(p=2, dim=0, keepdim=True))

            # create a decomposition of the weights matrix
            self._lora = LowRankAdapter(in_dim, out_dim, rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # combine the parameter matrices
        # shape (out_features, in_features)

        directional_component = (
            self._linear + self._alpha * self._lora.get_expanded_weights().T
        )

        # shape(1, in_features)
        normalization = torch.norm(directional_component, p=2, dim=0, keepdim=True)

        # "column"-wise normalization
        directional_component = directional_component / normalization

        # self.m has shape (1, in_features)
        # scaled_directional_component shape (out_featurs, in_features)
        scaled_directional_component = self.m * directional_component

        if self._bias is not None:
            return input @ scaled_directional_component.T + self._bias

        return input @ scaled_directional_component.T

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, rank: int, alpha: float = 1.0
    ) -> "WeightDecomposedLinearAdapater":
        return cls(linear.weight, rank, linear.bias, alpha)


_type_to_adapter_map: dict[str, LowRankLinearAdapterBase] = {
    "lora": LowRankLinearAdapter,
    "dora": WeightDecomposedLinearAdapater,
}


# find the parent of a nested attribute path
def _find_parent_of_nested_attr(obj, attribute_path: str):
    path = attribute_path.split(".")
    parent = obj
    for attr in path[:-1]:
        parent = getattr(parent, attr)
    
    if not hasattr(parent, path[-1]):
        raise AttributeError(f"Object {parent} does not have attribute {path[-1]}")
    
    return parent, path[-1]


def _set_module_attr(module: nn.Module, name: str, value: nn.Module) -> None:
    """Sets the attribute of a module with the specified name to the specified value.

    Args:
        module (nn.Module): module to set the attribute on.
        name (str): name of the attribute to set.
            Can be a "." separated nested attribute name, e.g. "adapter.lora.linear".
        value (nn.Module): value to set the attribute to.

    """
    parent, child_name = _find_parent_of_nested_attr(module, name)
    setattr(parent, child_name, value)
    

def adapt_model(
    model: nn.Module,
    low_rank_adapter_type: Literal["lora", "dora"],
    rank: int,
    alpha: float = 1,
) -> nn.Module:
    """Replaces all linear layers in the model with the specified low-rank adapter type in-place.

    Args:
        model (nn.Module): model to adapt.
        low_rank_adapter_type (Literal[&quot;lora&quot;, &quot;dora&quot;]): type of low-ranking adapter to use.
        rank (int): rank of the low-rank approximation.
        alpha (float, optional): strength of updates from the low-rank adapter. Defaults to 1.

    """
    if low_rank_adapter_type not in _type_to_adapter_map:
        raise ValueError(f"Invalid adapter type: {low_rank_adapter_type}")

    adapter_class = _type_to_adapter_map[low_rank_adapter_type]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _set_module_attr(model, name, adapter_class.from_linear(module, rank, alpha))

    return model


def merge_adapters(model: nn.Module) -> None:
    """Merges the weights of all low-rank adapters in the model with the original weights.

    Args:
        model (nn.Module): model with low-rank adapters to merge.

    """
    for module in model.modules():
        if isinstance(module, LowRankLinearAdapterBase):
            module.merge_weights()


class LowRankSequential(nn.Sequential):
    def _forward_no_split(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        for module in self:
            output = module(output)

        return output

    def _forward_with_split(
        self, input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_model_output = input
        lora_output = input

        for module in self:
            if isinstance(module, LowRankLinearAdapterBase):
                base_model_output, lora_output = module(
                    base_model_output, separate_lora_output=True
                )
            else:
                # pass both streams through the same module
                base_model_output = module(base_model_output)
                lora_output = module(lora_output)

        return base_model_output, lora_output

    def forward(
        self, input: torch.Tensor, separate_lora_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not separate_lora_output:
            return self._forward_no_split(input)

        return self._forward_with_split(input)

    @classmethod
    def from_sequential(
        cls, sequential: nn.Sequential, rank: int, alpha: float = 1.0
    ) -> "LowRankSequential":
        # adapt all layers in the sequential, and initialize the new sequential
        adapted_sequential = adapt_model(sequential, "lora", rank, alpha)
        return cls(*adapted_sequential.children())
