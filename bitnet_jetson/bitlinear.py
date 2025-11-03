from torch import nn, Tensor
import torch.nn.functional as F
import torch
from torch.utils import benchmark
from torch import nn

from torch.profiler import profile, record_function, ProfilerActivity
import ctypes
import numpy as np
# set all seed
torch.manual_seed(42)
np.random.seed(42)

bitnet_lib = ctypes.CDLL('kernels/bitnet_kernels/libbitnet.so')
def bitnet_int8xint2_linear(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()
    
    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.empty((1, M, N), dtype=torch.bfloat16, device=input0.device)
    bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])

    return ret
    

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
        self.out_features = out_features
        self.norm = nn.RMSNorm(in_features, eps=1e-5)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_quant', None)
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
        if(not self.training and self.quant):
            x_quant, x_scale = activation_norm_quant(x_norm)
            y = bitnet_int8xint2_linear(x_quant, self.weight_quant, x_scale, self.weight_scale)
            # print(x_scale[0,0,0])
            # y = bitnet_int8xint2_linear(x_quant, self.weight_quant, torch.ones_like(x_scale, device='cuda', dtype=torch.bfloat16), self.weight_scale)
            # y = bitnet_int8xint2_linear(x_quant, self.weight_quant, x_scale, torch.ones_like(self.weight_scale, device='cuda', dtype=torch.bfloat16))
            # y2 = bitnet_int8xint2_linear(x_quant, self.weight_quant, torch.ones_like(x_scale, device='cuda', dtype=torch.bfloat16), torch.ones_like(self.weight_scale, device='cuda', dtype=torch.bfloat16))
        elif(not self.training):
            w = self.weight
            x_quant, x_scale = activation_norm_quant(x_norm, True)
            y = F.linear(x_quant, w) / self.weight_scale / x_scale            
        else:
            w = self.weight
            # STE using detach
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            y = F.linear(x_quant, w_quant)

        return y
