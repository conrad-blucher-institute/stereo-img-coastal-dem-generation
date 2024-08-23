
import cv2
import os.path
import torch 
import numpy as np
import statistics as st 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False







def return_data_loaders(batch_size: int = 32, exp_name: str = 'test'):
    base_dir = '/home/hkaman/Documents/stereo-img-coastal-dem-generation/EXPs/' 
    save_dir = os.path.join(base_dir, f'EXP_{exp_name}')
    loss_dir = os.path.join(save_dir, 'loss')

    isExist = os.path.isdir(save_dir)

    if not isExist:
        os.mkdir(save_dir)
        os.mkdir(loss_dir)

    train_df = pd.read_csv('/home/hkaman/Documents/stereo-img-coastal-dem-generation/data/train_df.csv')
    valid_df = pd.read_csv('/home/hkaman/Documents/stereo-img-coastal-dem-generation/data/valid_df.csv')
    test_df  = pd.read_csv('/home/hkaman/Documents/stereo-img-coastal-dem-generation/data/test_df.csv')

    print(train_df.shape, valid_df.shape, test_df.shape)

    train_dataset = CustomDataset(train_df)
    valid_dataset = CustomDataset(valid_df)
    test_dataset  = CustomDataset(test_df)

    data_loader_training = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    data_loader_validate = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  
    data_loader_testing  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    return data_loader_training, data_loader_validate, data_loader_testing


class CustomDataset():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): 
        left_path = self.dataframe.loc[idx]['left_path']
        left_img = np.array(Image.open(left_path))
        left_img = left_img / 255.
        left_img = torch.as_tensor(left_img, dtype=torch.float32)

        right_path = self.dataframe.loc[idx]['right_path']
        right_img = np.array(Image.open(right_path))
        right_img = right_img / 255.
        right_img = torch.as_tensor(right_img, dtype=torch.float32)

        dem_path = self.dataframe.loc[idx]['dem_path']
        dem = np.array(Image.open(dem_path))
        dem = self.zero_interpolation(dem, method='nearest')
        dem = torch.as_tensor(dem, dtype=torch.float32)

        sample = {"left": left_img, "right": right_img, "dem": dem}

        return sample

    def zero_interpolation(self, image, method='nearest'):
        """
        Interpolate zero values in a single-channel image using neighboring non-zero values.
        
        Parameters:
            image (np.array): A 2D numpy array representing the single-channel image.
            method (str): Interpolation method; options are 'nearest', 'linear', 'cubic'.
        
        Returns:
            np.array: The interpolated image.
        """
        # Coordinates of known and unknown pixels
        y, x = np.indices(image.shape)
        known = image > 0
        known_x = x[known]
        known_y = y[known]
        known_values = image[known]

        # Grid on which to interpolate
        unknown = ~known
        unknown_x = x[unknown]
        unknown_y = y[unknown]

        # Perform interpolation using griddata
        filled_values = interpolate.griddata(
            (known_x, known_y),  # Coordinates of non-zero values
            known_values,        # Values at those coordinates
            (unknown_x, unknown_y),  # Coordinates where values are to be filled
            method=method  # Interpolation method
        )

        # Fill the original image array with interpolated values
        interpolated_image = image.copy()
        interpolated_image[unknown] = filled_values

        return interpolated_image
    