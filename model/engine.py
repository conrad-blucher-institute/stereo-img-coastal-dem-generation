
import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import random
from pathlib import Path
#======================================================================================================================================#
#====================================================== Training Config ===============================================================#
#======================================================================================================================================#   
seed = 1988
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"



class EarlyStopping():
    def __init__(self, tolerance=30, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, status):

        if status is True:
            self.counter = 0
        elif status is False: 
            self.counter +=1

        print(f"count: {self.counter}")
        if self.counter >= self.tolerance:  
                self.early_stop = True

def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves a model checkpoint during training.

    Parameters:
    - state (dict): State to save, including model and optimizer states.
    - filename (str): File name to save the checkpoint.
    """
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint into a model and optimizer.

    Parameters:
    - checkpoint_path (str): Path to the checkpoint file.
    - model (nn.Module): Model to load the checkpoint into.
    - optimizer (torch.optim): Optimizer to load the checkpoint into.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_val_loss']

class DEMPred:

    def __init__(self, model, lr: float, wd: float, exp: str):
        # Initialize model parameters and directories
        self.model = model
        self.lr = lr
        self.wd = wd
        self.exp = exp
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

        base_dir = Path('./EXPs')
        self.exp_output_dir = base_dir / f'EXP_{self.exp}'
        self.best_model_name = os.path.join(self.exp_output_dir, 'best_model_' + self.exp + '.pth')
        self.loss_fig_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.png')
        self.loss_df_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.csv')


    def mse_loss(self, inputs, targets):
        loss = (inputs - targets) ** 2
        loss = torch.mean(loss)
        
        return loss


    def train(self, data_loader_training, data_loader_validate, epochs: int, loss_stop_tolerance: int):
        """
        Trains the model using the provided training and validation data loaders.

        Parameters:
        - data_loader_training: DataLoader for the training dataset.
        - data_loader_validate: DataLoader for the validation dataset.
        - loss_type (str): Type of loss function to use ('mse', 'wmse', 'huber', 'wass').
        - epochs (int): Number of epochs to train the model.
        - loss_stop_tolerance (int): Early stopping tolerance level.
        """

        best_val_loss  = 1e15 # initial dummy value
        early_stopping = EarlyStopping(tolerance = loss_stop_tolerance, min_delta = 50)
        loss_stats = {'train': [],"val": []}
        

        for epoch in range(1, epochs + 1):

            training_start_time = time.time()
            train_epoch_loss = 0
            self.model.train()

            for batch, sample in enumerate(data_loader_training):
                
                left_img = sample['left'].to(device)
                right_img = sample['right'].to(device)
                dem_true = sample['dem'].to(device)

                # print(left_img.shape, right_img.shape, dem_true.shape)
                dem_pred = self.model(left_img, right_img) 

                self.optimizer.zero_grad()

                train_loss = self.mse_loss(dem_true, dem_pred)

                train_loss.backward()
                self.optimizer.step()
                train_epoch_loss += train_loss.item() 

            # VALIDATION    
            # self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_epoch_loss = 0
                for batch, sample in enumerate(data_loader_validate):
                    
                    l_img_val = sample['left'].to(device)
                    r_img_val = sample['right'].to(device)
                    dem_true_val = sample['dem'].to(device)

                    dem_pred_val = self.model(l_img_val, r_img_val)  #, _, _ 
                    valid_loss = self.mse_loss(dem_true_val, dem_pred_val)

                    val_epoch_loss += valid_loss.item()

            loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
            loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))

            training_duration_time = (time.time() - training_start_time)        
            print(f'Epoch {epoch+0:03}: | Time(s): {training_duration_time:.3f}| Train Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 


            if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                        
                best_val_loss=(val_epoch_loss/len(data_loader_validate))
                torch.save(self.model.state_dict(), self.best_model_name)
                print(f'=============================== Best model Saved! Val MSE: {best_val_loss:.4f}')

                status = True
            else:

                status = False

            early_stopping(status)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch:", epoch)
                break

        save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)


