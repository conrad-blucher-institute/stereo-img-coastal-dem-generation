# import sys
# sys.path.append('../')
# from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
import random
import torch
from torch import nn
import os 
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from model.attention import RegressionHead, PatchEmbedding, SpatialEncoder
device = "cuda" if torch.cuda.is_available() else "cpu"

class DemViT(nn.Module):
    def __init__(self, ): 
        super().__init__()

                
        # model_configs = {
        #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        # }

        # encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

        # self.model = DepthAnythingV2(**model_configs[encoder])
        # self.model.load_state_dict(torch.load(f'/data1/fog/stereo-img-coastal-dem-generation/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        # self.model = self.model.to(device).eval()

        self.img_embeds = PatchEmbedding(img_size = (640, 240), patch_size=(80, 80), emb_size=768, dropout_rate=0.1)

        self.encoder = SpatialEncoder(dim = 768, depth = 8, heads = 8, dim_head = 96, mult=4, dropout=0.1)
        
        self.head = RegressionHead(embed_dim = 768, img_size = (150, 120))

    # def depth_process(self, img):
    #     # If img is batched, process each image in the batch
    #     if len(img.shape) == 4:  # Shape is (batch_size, h, w, 3)
    #         depths = []
    #         for i in range(img.shape[0]):  # Iterate over the batch
    #             depth = self.model.infer_image(img[i])
    #             depth = torch.unsqueeze(torch.tensor(depth), dim=0)  # Add batch dimension back
    #             depths.append(depth)
    #         depth = torch.cat(depths, dim=0)  # Concatenate along the batch dimension
    #     else:
    #         # If not batched, directly process the image
    #         depth = self.model.infer_image(img)
    #         depth = torch.unsqueeze(torch.tensor(depth), dim=0)  # Add batch dimension back
        
    #     # Ensure the depth has a channel dimension
    #     depth = torch.unsqueeze(depth, dim=1)
        
    #     return depth
    
    
    def forward(self, 
                left: torch.Tensor,
                right: torch.Tensor): 

        left = left.permute(0, 3, 1, 2)
        right = right.permute(0, 3, 1, 2)
        
        # left_depth = self.depth_process(left)
        # right_depth = self.depth_process(right)

        img = self.img_embeds(left, right)
        img = self.encoder(img)

        # img = img.mean(dim=1)
        preds = self.head(img)

        return preds