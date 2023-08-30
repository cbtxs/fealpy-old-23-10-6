
from typing import Sequence
import math

import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter


class Standardize(Module):
    """
    @brief: Standardize the inputs with sequence of centers and radius.\
    Stack the result with shape like (#Samples, #Centers, #Dims).
    """
    def __init__(self, centers: Tensor, radius: Tensor, device=None):
        """
        @param centers: Tensor with shape (M, GD).
        @param radius: Tensor with shape (M,).

        @note:
        If shape of center and radius is (M, D) and shape of input is (N, D),
        then the shape of output is (N, M, D); where N is samples, M is number of
        centers to standardize and D is features.
        """
        super().__init__()
        self.centers = Parameter(centers.to(device=device), requires_grad=False)
        self.radius = Parameter(radius.to(device=device), requires_grad=False)

    def forward(self, p: Tensor):
        return (p[:, None, :] - self.centers[None, :, :]) / self.radius[None, :, None]


class Distance(Module):
    """
    @brief Calculate the distances between inputs and source points.\
    Return with shape like (#Samples, #Sources).
    """
    def __init__(self, sources: Tensor, device=None) -> None:
        """
        @param sources: Tensor with shape (#Sources, #Dims).
        """
        super().__init__()
        self.sources = Parameter(sources.to(device=device), requires_grad=False)

    def forward(self, p: Tensor):
        return torch.norm(p[:, None, :] - self.sources[None, :, :], p=2,
                          dim=-1, keepdim=False)

    def gradient(self, p: Tensor):
        """
        @brief Return the gradient with shape (#Samples, #Sources, #Dims).
        """
        dis = self.forward(p)[:, :, None]
        return (p[:, None, :] - self.sources[None, :, :]) / dis


class MultiLinear(Module):
    def __init__(self, in_features: int, out_features: int, parallel: Sequence[int],
                 bias: bool=True, device=None, dtype=None, requires_grad=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.ni = in_features
        self.no = out_features
        self.p = tuple(parallel)
        self.weight = Parameter(
            torch.empty(self.p + (self.ni, self.no), **factory_kwargs),
            requires_grad=requires_grad
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.p + (self.no, ), **factory_kwargs),
                requires_grad=requires_grad
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # fan_in = self.weight.shape[-2]
        # gain = init.calculate_gain('leaky_relu', math.sqrt(5))
        # std = gain / math.sqrt(fan_in)
        # bound = math.sqrt(3.0) * std
        bound = 1
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            # bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        ret = torch.einsum('...io, n...i -> n...o', self.weight, x)
        return self.bias[None, ...] + ret
