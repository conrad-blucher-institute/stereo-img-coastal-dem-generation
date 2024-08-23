
import cv2
import os.path
import torch 
import numpy as np
import statistics as st 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader



def return_data_loaders(batch_size: int = 32, exp_name: str = 'test'):
    save_dir = '/data1/fog/stereo-img-coastal-dem-generation/EXPs/' + exp_name
    isExist = os.path.isdir(save_dir)

    if not isExist:
        os.mkdir(save_dir)

    train_df = pd.read_csv('/data1/fog/stereo-img-coastal-dem-generation/data/train_df.csv')
    valid_df = pd.read_csv('/data1/fog/stereo-img-coastal-dem-generation/data/valid_df.csv')
    test_df  = pd.read_csv('/data1/fog/stereo-img-coastal-dem-generation/data/test_df.csv')

    print(train_df.shape, valid_df.shape, test_df.shape)

    train_dataset = CustomDataset(train_df)
    valid_dataset = CustomDataset(valid_df)
    test_dataset  = CustomDataset(test_df)

    data_loader_training = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    data_loader_validate = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  
    data_loader_testing  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    return data_loader_training, data_loader_validate, data_loader_testing


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): 
        left_path = self.dataframe.loc[idx]['left_path']
        left_img = np.array(Image.open(left_path))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0
        left_img = torch.as_tensor(left_img, dtype=torch.float32)

        right_path = self.dataframe.loc[idx]['right_path']
        right_img = np.array(Image.open(right_path))
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) / 255.0
        right_img = torch.as_tensor(right_img, dtype=torch.float32)

        dem_path = self.dataframe.loc[idx]['dem_path']
        dem = np.array(Image.open(dem_path))
        dem = torch.as_tensor(dem, dtype=torch.float32)

        sample = {"left": left_img, "right": right_img, "dem": dem}

        return sample
