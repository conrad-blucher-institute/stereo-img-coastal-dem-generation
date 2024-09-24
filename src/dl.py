
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
import os
from datetime import datetime
import glob
from collections import defaultdict
from pathlib import Path
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def return_data_loaders(batch_size: int = 32, exp_name: str = 'test', strategy: str = 'seasonal'):

    # Define the base directory relative to the current directory
    base_dir = '../stereo-img-coastal-dem-generation/EXPs'

    # Construct save directory and loss directory paths
    # save_dir = 'C:\Users\mhajiesmaeeli\OneDrive - Texas A&M University-Corpus Christi\Documents\Mona\course\Thesis\GitHub\stereo-img-coastal-dem-generation\EXPs\EXP_003' #base_dir / f'EXP_{exp_name}'
    # loss_dir = 'C:\Users\mhajiesmaeeli\OneDrive - Texas A&M University-Corpus Christi\Documents\Mona\course\Thesis\GitHub\stereo-img-coastal-dem-generation\EXPs\EXP_003\loss' #save_dir / 'loss'

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_dir.mkdir(parents=True, exist_ok=True)

    # Define the left, right, and dem directories using pathlib for consistency
    # left_dir = 'C:\Users\mhajiesmaeeli\OneDrive - Texas A&M University-Corpus Christi\Documents\Mona\course\Thesis\GitHub\stereo-img-coastal-dem-generation\data\left\' #'../stereo-img-coastal-dem-generation/data/left/'
    # right_dir = 'C:\Users\mhajiesmaeeli\OneDrive - Texas A&M University-Corpus Christi\Documents\Mona\course\Thesis\GitHub\stereo-img-coastal-dem-generation\data\right\'    #'../stereo-img-coastal-dem-generation/data/right/'
    # dem_dir = 'C:\Users\mhajiesmaeeli\OneDrive - Texas A&M University-Corpus Christi\Documents\Mona\course\Thesis\GitHub\stereo-img-coastal-dem-generation\data\dem\'   #'../stereo-img-coastal-dem-generation/data/dem/'
    # # Define the base directory
    # base_dir = Path('./stereo-img-coastal-dem-generation/EXPs')

    # # Construct save directory and loss directory paths
    # save_dir = base_dir / f'EXP_{exp_name}'
    # loss_dir = save_dir / 'loss'

    # # Check if the save directory exists, and if not, create it along with the loss directory
    # if not save_dir.exists():
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     loss_dir.mkdir(parents=True, exist_ok=True)

    # # Define the left, right, and dem directories using pathlib for consistency
    # left_dir = Path('./stereo-img-coastal-dem-generation/data/left/') 
    # right_dir = Path('./stereo-img-coastal-dem-generation/data/right/')
    # dem_dir = Path('./stereo-img-coastal-dem-generation/data/dem/')

    train_df, valid_df, test_df = SplitData(left_dir, right_dir, dem_dir, strategy = strategy).create_dfs()

    # train_df = pd.read_csv('/home/hkaman/Documents/stereo-img-coastal-dem-generation/data/train_df.csv')
    # valid_df = pd.read_csv('/home/hkaman/Documents/stereo-img-coastal-dem-generation/data/valid_df.csv')
    # test_df  = pd.read_csv('/home/hkaman/Documents/stereo-img-coastal-dem-generation/data/test_df.csv')

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

class SplitData():
    def __init__(self, left_dir, right_dir, dem_dir, strategy: str):
        self.strategy = strategy
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.dem_dir = dem_dir
        self.left_files = sorted(glob.glob(os.path.join(left_dir, '*.tiff')))
        self.right_files = sorted(glob.glob(os.path.join(right_dir, '*.tiff')))
        self.dem_files = sorted(glob.glob(os.path.join(dem_dir, '*.tif')))
    
    def create_dfs(self):
        if self.strategy == 'daily':
            train, valid, test = self.daily_strategy()
        elif self.strategy == 'timeseries':
            train, valid, test = self.timeseries_strategy()
        elif self.strategy == 'seasonal':
            train, valid, test = self.date_strategy()

        return train, valid, test

    def daily_strategy(self):

        # Dictionary to store lists of files by date
        data_by_date = defaultdict(list)
        
        # Extract date from DEM filenames and group files by date
        for left_file in self.left_files:
            left_name = os.path.basename(left_file)
            date = left_name[1:-7]  
            data_by_date[date].append(left_name)
        
        train_data = []
        valid_data = []
        test_data = []
        
        # Distribute files into train, validation, and test sets
        for date, left_files in data_by_date.items():

            if len(left_files) < 3:
                train_data.append(left_files.pop())

            elif len(left_files) == 3:
                random.shuffle(left_files)
                valid_data.append(left_files[0])
                test_data.append(left_files[1])
                train_data.append(left_files[2])
            elif len(left_files) > 3:
                random.shuffle(left_files)
                valid_data.append(left_files.pop())
                test_data.append(left_files.pop())
                train_data.extend(left_files)
            else:
                raise ValueError(f"Unexpected number of DEM files for date {date}: {len(self.dem_files)}")
        

        # Create the dataframes
        def create_dataframe(left_files):
            rows = []
            for left_file in left_files:
                date = os.path.basename(left_file)[1:-7]
                time_part = os.path.basename(left_file)[9:11]

                dem_file = os.path.join(self.dem_dir, f"DEM{date}.tif")
                right_file = os.path.join(self.right_dir, f"R{date}{time_part}.tiff")
                left_file = os.path.join(self.left_dir, f"L{date}{time_part}.tiff")
                
                if os.path.exists(left_file) and os.path.exists(right_file):
                    rows.append({
                        'date': date,
                        'left_path': left_file,
                        'right_path': right_file,
                        'dem_path': dem_file
                    })
            
            return pd.DataFrame(rows)

        # print(len(train_data), len(valid_data), len(test_data))
        train_df = create_dataframe(train_data)
        valid_df = create_dataframe(valid_data)
        test_df = create_dataframe(test_data)
        
        return train_df, valid_df, test_df
    
    def timeseries_strategy(self):

    
        # Dictionary to store lists of files by date
        data_by_date = defaultdict(list)
        
        # Extract date from DEM filenames and group files by date
        for left_file in self.left_files:
            left_name = os.path.basename(left_file)
            date = left_name[1:-7]  # Extract the date part from the filename
            data_by_date[date].append(left_name)
        
        # Sort the dates
        # sorted_dates = sorted(data_by_date.keys())
        sorted_dates = sorted(data_by_date.keys(), key=lambda x: datetime.strptime(x, '%m%d%Y'))
    

        # Split into train, validation, and test sets
        train_dates = sorted_dates[:7]  # First 7 dates for training
        valid_dates = sorted_dates[7:9]  # Next 2 dates for validation
        test_dates = sorted_dates[9:]  # Last 2 dates for testing

        train_data = []
        valid_data = []
        test_data = []
        
        # Distribute files into train, validation, and test sets
        for date, left_files in data_by_date.items():
            if date in train_dates:
                train_data.extend(left_files)
            elif date in valid_dates:
                valid_data.extend(left_files)
            elif date in test_dates:
                test_data.extend(left_files)

        # Function to create dataframes
        def create_dataframe(left_files):
            rows = []
            for left_file in left_files:
                date = os.path.basename(left_file)[1:-7]
                time_part = os.path.basename(left_file)[9:11]

                dem_file = os.path.join(self.dem_dir, f"DEM{date}.tif")
                right_file = os.path.join(self.right_dir, f"R{date}{time_part}.tiff")
                left_file = os.path.join(self.left_dir, f"L{date}{time_part}.tiff")
                
                if os.path.exists(left_file) and os.path.exists(right_file):
                    rows.append({
                        'date': date,
                        'left_path': left_file,
                        'right_path': right_file,
                        'dem_path': dem_file
                    })
            
            return pd.DataFrame(rows)

        # Create dataframes for train, validation, and test sets
        train_df = create_dataframe(train_data)
        valid_df = create_dataframe(valid_data)
        test_df = create_dataframe(test_data)
        
        return train_df, valid_df, test_df
    
    def date_strategy(self):

        train_indices = [0, 2, 4, 6, 8, 9, 11]
        valid_indices = [1, 5, 10]
        test_indices = [3, 7]

        data_by_date = defaultdict(list)
        
        # Extract date from DEM filenames and group files by date
        for left_file in self.left_files:
            left_name = os.path.basename(left_file)
            date = left_name[1:-7]  # Extract the date part from the filename
            data_by_date[date].append(left_name)
        
        # Sort the dates
        sorted_dates = sorted(data_by_date.keys(), key=lambda x: datetime.strptime(x, '%m%d%Y'))
        print(sorted_dates)
        
        # Split into train, validation, and test sets using given indices
        train_dates = [sorted_dates[i] for i in train_indices]
        valid_dates = [sorted_dates[i] for i in valid_indices]
        test_dates = [sorted_dates[i] for i in test_indices]

        train_data = []
        valid_data = []
        test_data = []
        
        # Distribute files into train, validation, and test sets
        for date, left_files in data_by_date.items():
            if date in train_dates:
                train_data.extend(left_files)
            elif date in valid_dates:
                valid_data.extend(left_files)
            elif date in test_dates:
                test_data.extend(left_files)

        # Function to create dataframes
        def create_dataframe(left_files):
            rows = []
            for left_file in left_files:
                date = os.path.basename(left_file)[1:-7]
                time_part = os.path.basename(left_file)[9:11]

                dem_file = os.path.join(self.dem_dir, f"DEM{date}.tif")
                right_file = os.path.join(self.right_dir, f"R{date}{time_part}.tiff")
                left_file = os.path.join(self.left_dir, f"L{date}{time_part}.tiff")
                
                if os.path.exists(left_file) and os.path.exists(right_file):
                    rows.append({
                        'date': date,
                        'left_path': left_file,
                        'right_path': right_file,
                        'dem_path': dem_file
                    })
            
            return pd.DataFrame(rows)

        # Create dataframes for train, validation, and test sets
        train_df = create_dataframe(train_data)
        valid_df = create_dataframe(valid_data)
        test_df = create_dataframe(test_data)
        
        return train_df, valid_df, test_df

