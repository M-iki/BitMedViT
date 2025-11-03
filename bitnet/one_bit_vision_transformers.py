import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from bitnet.bitlinear import BitLinear
from zeta.nn import MultiheadAttention, MultiQueryAttention
from timm.layers import PatchEmbed
from timm.layers import DropPath

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(dim, hidden_dim),
            nn.GELU(),
            BitLinear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

def replace_linear_layers(model, custom_linear_class):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with your custom layer, preserving dimensions
            new_module = custom_linear_class(module.in_features, module.out_features, bias=module.bias is not None)
            setattr(model, name, new_module)
        else:
            # Recurse into submodules
            replace_linear_layers(module, custom_linear_class)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, multi_query = False, full_precision_attn = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # module = nn.ModuleList(
            #         [MultiQueryAttention(dim, heads), 
            #          FeedForward(dim, mlp_dim), 
            #          nn.LayerNorm(dim), 
            #          nn.LayerNorm(dim)]
            #     )
            module = nn.ModuleList(
                [
                    MultiQueryAttention(dim, heads) if multi_query else MultiheadAttention(dim, heads), 
                    nn.Dropout(0.1),
                    DropPath(0.1),
                    FeedForward(dim, mlp_dim), 
                    nn.Dropout(0.1),
                    DropPath(0.1),
                    nn.LayerNorm(dim) if full_precision_attn else nn.Identity(), 
                    nn.Identity()
                ]
            )
            for name, submodule in module.named_modules():
                if submodule.__class__.__name__ == "MultiwayNetwork":
                    # Get parent and attribute name
                    parent_name = ".".join(name.split(".")[:-1])
                    attr_name = name.split(".")[-1]

                    # Navigate to parent module
                    parent = module
                    if parent_name:
                        for attr in parent_name.split("."):
                            parent = getattr(parent, attr)

                    # Replace the MultiwayNetwork with its 'A' submodule
                    setattr(parent, attr_name, submodule.A)
            if(not full_precision_attn):
                replace_linear_layers(module, BitLinear)

            self.layers.append(
                module    
            )
            self.multi_query = multi_query

    def forward(self, x):
        layers = []
        for attn, drop1, droppath1, ff, drop2, droppath2, norm1, norm2 in self.layers:
            x_norm = norm1(x)
            if(self.multi_query):
                x_attn, _, _ = attn(x_norm)
            else:
                x_attn = attn(x_norm, x_norm, x_norm)
            x = droppath1(drop1(x_attn)) + x
            x = droppath2(drop2(ff(norm2(x)))) + x
            layers.append(x)
        return x, layers


class OneBitViT(nn.Module):
    """
    OneBitViT is a vision transformer model for image classification tasks.

    Args:
        image_size (int or tuple): The size of the input image. If an integer is provided, it is assumed to be a square image.
        patch_size (int or tuple): The size of each patch in the image. If an integer is provided, it is assumed to be a square patch.
        num_classes (int): The number of output classes.
        dim (int): The dimensionality of the token embeddings and the positional embeddings.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads in the transformer.
        mlp_dim (int): The dimensionality of the feed-forward network in the transformer.
        channels (int): The number of input channels in the image. Default is 3.
        dim_head (int): The dimensionality of each attention head. Default is 64.

    Attributes:
        to_patch_embedding (nn.Sequential): Sequential module for converting image patches to embeddings.
        pos_embedding (torch.Tensor): Positional embeddings for the patches.
        transformer (Transformer): Transformer module for processing the embeddings.
        pool (str): Pooling method used to aggregate the patch embeddings. Default is "mean".
        to_latent (nn.Identity): Identity module for converting the transformer output to the final latent representation.
        linear_head (nn.LayerNorm): Layer normalization module for the final linear projection.

    Methods:
        forward(img): Performs a forward pass through the OneBitViT model.

    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        matcher_dim=49,
        multi_query=False,
        full_precision_attn = False
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.gradcam = False
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # BitLinear(patch_dim, dim),
            # nn.LayerNorm(dim),
        )
        # self.to_patch_embedding = PatchEmbed(
        #     img_size=image_size,
        #     patch_size=patch_size,
        #     in_chans=channels,
        #     embed_dim=dim,
        #     bias=True,  # disable bias if pre-norm is used (e.g. CLIP)
        #     dynamic_img_pad=False
        # )

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.pos_embedding = posemb_sincos_2d(
        #     h=image_height // patch_height,
        #     w=image_width // patch_width,
        #     dim=dim,
        # )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, multi_query, full_precision_attn)

        # self.norm = nn.LayerNorm(dim)

        self.classifier = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        
        self.matcher = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, matcher_dim)
        )        
    def forward(self, img):
        device = img.device
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x, layers = self.transformer(x)
        if(self.training):
            match = self.matcher(layers[-1])
        
        x = x.mean(dim=1)
        out = self.classifier(x)
        
        if(self.gradcam):
            return out
        if(not self.training):
            return out        
        else:
            return out, match


# import torch
# from bitnet import SimpleViT

# v = OneBitViT(
#     image_size=256,
#     patch_size=32,
#     num_classes=1000,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
# )

# img = torch.randn(1, 3, 256, 256)

# preds = v(img)  # (1, 1000)
# print(preds)
