import torch
from torch import nn, Tensor
import torch.nn.functional as F


def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def activation_norm_quant(x: Tensor, bfloat = False):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp_(-128, 127).to(torch.bfloat16 if bfloat else torch.int8), scale.to(torch.bfloat16)


def weight_quant(w: Tensor):
    # scale = w.abs().mean()
    # e = w.mean()
    # u = (w - e).sign() * scale

    scale = 1.0 / (w.abs().mean().clamp_(min=1e-5))
    u=(w*scale).round().clamp(-1, 1) / scale
    return u

def weight_inference_quant(w: Tensor):
    # scale = w.abs().mean()
    # e = w.mean()
    # u = (w - e).sign() * scale

    scale = 1.0 / (w.abs().mean().clamp_(min=1e-5))
    u=(w*scale).round().clamp(-1, 1)
    return u, scale

class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.norm = nn.RMSNorm(in_features, eps=1e-5)
        self.register_buffer('weight_scale', None)
        self.quant = False
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x_norm = self.norm(x)

        if(not self.training):
            w = self.weight
            x_quant, x_scale = activation_norm_quant(x_norm, True)
            y = F.linear(x_quant, w.to(torch.bfloat16)) / self.weight_scale / x_scale            
        else:
            w = self.weight
            # STE using detach
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            y = F.linear(x_quant, w_quant)
        return y
