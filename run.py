import os
# from model import depth_vit
import torch
import argparse
import numpy as np
import random
from src import dl
from model import engine

from model.depth_vit import DemViT

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    Exp_name = args.exp_name
    batch_size = args.batch_size
    embd_size = args.embd_size
    num_heads = args.num_heads
    num_layers = args.num_layers
    in_channels = args.in_channels
    dropout = args.dropout
    post_norm = args.postnorm
    lr = args.lr
    wd = args.wd
    epochs = args.epochs

    data_loader_training, data_loader_validate, data_loader_test = dl.return_data_loaders(
        batch_size = batch_size, 
        exp_name = Exp_name, 
        strategy= 'timeseries'
        )

    model = DemViT().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")


    YE = engine.DEMPred(model, 
                        lr = lr, 
                          wd = wd, 
                          exp = Exp_name)

    _ = YE.train(data_loader_training, data_loader_validate, 
                      epochs = epochs, 
                      loss_stop_tolerance = 200)


    # _ = YE.predict(model, data_loader_training, category= 'train', iter = 1)
    # _ = YE.predict(model, data_loader_validate, category= 'valid', iter = 1)
    # _ = YE.predict(model, data_loader_test, category= 'test', iter = 1)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name",    type=str,   default = "004",help = "Experiment name")
    parser.add_argument("--batch_size",  type=int,   default = 64,   help = "Batch size")
    parser.add_argument("--embd_size",   type=int,   default = 768,  help = "Embedding size")
    parser.add_argument("--num_heads",   type=int,   default = 8,     help = "Number of attention heads")
    parser.add_argument("--num_layers",  type=int,   default = 6,     help = "Number of transformer layers")
    parser.add_argument("--in_channels", type=int,   default = 8,     help = "Number of input channels")
    parser.add_argument("--dropout",     type=float, default = 0.1,   help = "Amount of dropout")
    parser.add_argument("--postnorm",    type=str,   default = False, help = "Post or Before Normalization for Self-Attention")
    parser.add_argument("--lr",          type=float, default = 0.0001, help = "Learning rate")
    parser.add_argument("--wd",          type=float, default = 0.01,  help = "Value of weight decay")
    parser.add_argument("--epochs",      type=int,   default = 200,   help = "The number of epochs")

    args = parser.parse_args()

    main(args)