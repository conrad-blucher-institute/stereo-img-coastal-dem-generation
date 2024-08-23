import torch
import torch.nn as nn
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn.modules.utils import _pair
import os 
from typing import Dict, Union
from torch.nn.functional import interpolate
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class RegressionHead(nn.Module):

    def __init__(self, embed_dim, img_size = (150, 120)):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

        # reshape 
        self.norm =nn.LayerNorm(embed_dim)
        self.lfc = nn.Linear(self.embed_dim, 360, bias=True)  
        self.fcn = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.lfc(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], self.img_size[0], self.img_size[1])

        return x

def default(val, d):
    return val if val is not None else d

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if not glu:
            self.project_in = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU()
            )
        else:
            self.project_in = GEGLU(dim, inner_dim)  
        
        self.dropout = nn.Dropout(dropout)
        self.project_out = nn.Linear(inner_dim, dim_out)
        
        self.net = nn.Sequential(
            self.project_in,
            self.dropout,
            self.project_out
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if isinstance(x, list):
            # Apply normalization and function fn to each tensor in the list
            return [self.fn(self.norm(xi)) for xi in x]
        else:
            # Single tensor processing as usual
            return self.fn(self.norm(x), **kwargs)
          
class PatchEmbedding(nn.Module):
    def __init__(self, img_size = (640, 240), patch_size=(80, 80), emb_size=768, dropout_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        

        h_patches = self.img_size[0] // self.patch_size[0]
        w_patches = self.img_size[1] // self.patch_size[1]
        self.n_patches = h_patches * w_patches
        

        self.patch_embeddings = nn.Conv2d(in_channels=3, out_channels=emb_size,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, (self.num_patches*2) + 2, emb_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(emb_size)
        
    def forward(self, x_left, x_right):
        # Process left and right images
        def process_image(x):
            # x: (B, 3, H, W)
            x = self.patch_embeddings(x)  # (B, emb_size, h_patches, w_patches)
            x = x.flatten(2)              # (B, emb_size, n_patches)
            x = x.transpose(1, 2)         # (B, n_patches, emb_size)
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, emb_size)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+n_patches, emb_size)
            return x
        
        x_left = process_image(x_left)
        x_right = process_image(x_right)
        
        # Concatenate embeddings from both images
        x = torch.cat((x_left, x_right), dim=1)  # (B, 2*(1+n_patches), emb_size)
        
        # Adding position embeddings (assuming same position for both images)
        positions = self.position_embeddings[:, :x.size(1)]
        x = x + positions

        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x

class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SpatialEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)





# class SpatialAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         attn = dots.softmax(dim=-1)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out

# class SpatialEncoder(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.norm = nn.LayerNorm(dim)
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return self.norm(x)