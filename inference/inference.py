import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import interpolate
import rasterio
device = "cuda" if torch.cuda.is_available() else "cpu"
from scipy.ndimage import gaussian_filter
import sys
sys.path.append('../')
from model.depth_vit import DemViT
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
seed = 42
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PLOT_CMAP = 'viridis'
PLOT_MINCNT = 100
PLOT_VIS_FACE_COLOR = 'white'
PLOT_BOX_FACE_COLOR = 'grey'
MAXIMUM_AXIS_VALUE = 2

import matplotlib.gridspec as gridspec
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())



def predict(data_loader, exp_name: str):

    exp_output_dir = r"C:\Users\mhajiesmaeeli\OneDrive - Texas A&M University-Corpus Christi\Documents\Mona\course\Thesis\GitHub\stereo-img-coastal-dem-generation\EXPs" + 'EXP_' + exp_name
    best_model_name = os.path.join(exp_output_dir, 'best_model_' + exp_name + '.pth')

    model = DemViT().to(device)
    model.load_state_dict(torch.load(best_model_name))


    output_files = []
    with torch.no_grad():
        data_dict = {}
        for sample in data_loader:
            l_img = sample['left'].to(device)
            r_img = sample['right'].to(device)
            ytrue = sample['dem'].detach().cpu().numpy()


            ypred = model(l_img, r_img) 

            this_batch = {"left": l_img.detach().cpu().numpy(), 
                        "right": r_img.detach().cpu().numpy(), 
                        "true": ytrue,
                        "pred": ypred.detach().cpu().numpy()
                        }
            
            output_files.append(this_batch)

    return output_files

def zero_interpolation(image, method='nearest'):
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

def scatter_plot(train, val, test):
    # Function to plot data
    def plot_data(data, title, ax):
        # Apply zero interpolation and concatenate true and predicted values
        sigma = 4  # Standard deviation of the Gaussian kernel, adjust as needed
        true_values = np.concatenate([zero_interpolation(d['true'][0, ...]).flatten() for d in data]).flatten()
        pred_values = np.concatenate([gaussian_filter(zero_interpolation(d['pred'][0, ...]), sigma = sigma).flatten() for d in data]).flatten()
        
        # Calculate RMSE and R2
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        r2 = r2_score(true_values, pred_values)

        # Scatter plot
        ax.scatter(true_values, pred_values, alpha=0.5)
        ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2)
        ax.set_title(f'{title}\nRMSE: {rmse:.3f}, R^2: {r2:.3f}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plot_data(train, 'Train', axs[0])
    plot_data(val, 'Validation', axs[1])
    plot_data(test, 'Test', axs[2])

    plt.tight_layout()
    plt.show()

def _plot_scatter(data):

    sigma = 4  # Standard deviation of the Gaussian kernel, adjust as needed
    true_values = np.concatenate([zero_interpolation(d['true'][0, ...]).flatten() for d in data]).flatten()
    pred_values = np.concatenate([gaussian_filter(zero_interpolation(d['pred'][0, ...]), sigma = sigma).flatten() for d in data]).flatten()
    
    # Calculate RMSE and R2
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    r2 = r2_score(true_values, pred_values)

    g = sns.jointplot(x = true_values, 
                        y = pred_values, 
                        kind = "hex", 
                        height = 8, 
                        ratio = 4,
                        xlim = [0.5, MAXIMUM_AXIS_VALUE], ylim=[0.5, MAXIMUM_AXIS_VALUE], 
                        extent = [0.5, MAXIMUM_AXIS_VALUE, 0.5, MAXIMUM_AXIS_VALUE], 
                        gridsize = 100,
                        cmap = PLOT_CMAP, 
                        mincnt = PLOT_MINCNT, 
                        joint_kws = {"facecolor": PLOT_VIS_FACE_COLOR})

    for patch in g.ax_marg_x.patches:
        patch.set_facecolor(PLOT_BOX_FACE_COLOR)

    for patch in g.ax_marg_y.patches:
        patch.set_facecolor(PLOT_BOX_FACE_COLOR)

    g.ax_joint.plot([0.5, MAXIMUM_AXIS_VALUE], [0.5, MAXIMUM_AXIS_VALUE], '--r', linewidth=2)

    plt.xlabel('Measured DEM (m)', fontsize = 16)
    plt.ylabel('Predicted DEM (m)', fontsize = 16)
    plt.grid(False)

    scores = (r'R^2={:.2f}' + '\n' + r'RMSE={:.3f} (m)').format(
        r2, rmse)

    plt.text(0.6, 2, scores, bbox=dict(facecolor = PLOT_VIS_FACE_COLOR, edgecolor = PLOT_BOX_FACE_COLOR, boxstyle = 'round, pad=0.2'),
            fontsize = 16, ha='left', va = 'top')

    return g



def joinplot_hex_vis(train, val, test):
    # Function to plot data
    fig = plt.figure(figsize=(21, 7))
    gs  = gridspec.GridSpec(1, 3)

    train_plot = _plot_scatter(train)
    valid_plot = _plot_scatter(val)
    test_plot = _plot_scatter(test)

    mg0 = SeabornFig2Grid(train_plot, fig, gs[0])
    mg1 = SeabornFig2Grid(valid_plot, fig, gs[1])
    mg2 = SeabornFig2Grid(test_plot, fig, gs[2])

    gs.tight_layout(fig)
    plt.show()

def plot_sample_by_index(data, index):
    # Retrieve the specific item by index
    sample = data[index]
    
    # Unpack the left, right, true, and pred from the sample
    left_img = sample['left'][0, ...]
    right_img = sample['right'][0, ...]
    true_dem = sample['true'][0, ...]
    pred_dem = sample['pred'][0, ...]
    pred_dem = zero_interpolation(pred_dem, method='nearest')
    sigma = 4  # Standard deviation of the Gaussian kernel, adjust as needed
    pred_dem = gaussian_filter(pred_dem, sigma=sigma)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot Left Image
    axs[0].imshow(left_img)
    axs[0].set_title('Left Image')
    axs[0].axis('off')  # Turn off axis
    
    # Plot Right Image
    axs[1].imshow(right_img)
    axs[1].set_title('Right Image')
    axs[1].axis('off')  # Turn off axis
    
    # # Plot True DEM with fixed color limits
    # dem1 = axs[2].imshow(true_dem, cmap='viridis', vmin=1, vmax=1.5)
    # axs[2].set_title('Measured DEM (m)')
    # axs[2].axis('off')  # Turn off axis
    # cbar1 = plt.colorbar(dem1, ax=axs[2], fraction=0.046, pad=0.04, shrink = 1.5)
    # cbar1.set_label('Elevation [NAVD88] (m)')
    
    # # Plot Predicted DEM with fixed color limits
    # dem2 = axs[3].imshow(pred_dem, cmap='viridis', vmin=1, vmax=1.5)
    # axs[3].set_title('Predicted DEM (m)')
    # axs[3].axis('off')  # Turn off axis
    # cbar2 = plt.colorbar(dem2, ax=axs[3], fraction=0.046, pad=0.04, shrink = 1.5)
    # cbar2.set_label('Elevation [NAVD88] (m)')

    # # Create a divider for the existing axes instance
    # divider2 = make_axes_locatable(axs[3])
    # # Append a new axes to the right of axs[3], for the color bar
    # cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    # cbar2 = plt.colorbar(dem2, cax=cax2)
    # cbar2.set_label('Elevation [NAVD88] (m)')
    # Plot True DEM with fixed color limits
    dem1 = axs[2].imshow(true_dem, cmap='viridis', vmin=1, vmax=1.5)
    axs[2].set_title('Measured DEM (m)')
    axs[2].axis('off')  # Turn off axis

    # Create a divider for the existing axes instance
    divider1 = make_axes_locatable(axs[2])
    # Append a new axes to the right of axs[2], for the color bar
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(dem1, cax=cax1)
    cbar1.set_label('Elevation [NAVD88] (m)')

    # Plot Predicted DEM with fixed color limits
    dem2 = axs[3].imshow(pred_dem, cmap='viridis', vmin=1, vmax=1.5)
    axs[3].set_title('Predicted DEM (m)')
    axs[3].axis('off')  # Turn off axis

    # Create a divider for the existing axes instance
    divider2 = make_axes_locatable(axs[3])
    # Append a new axes to the right of axs[3], for the color bar
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(dem2, cax=cax2)
    cbar2.set_label('Elevation [NAVD88] (m)')
    plt.show()

def plot_tif_images_and_histograms(folder_path):
    """
    Reads .tif files from a specified folder, and plots each image next to a histogram of its values.
    
    Parameters:
        folder_path (str): Path to the folder containing .tif files.
    """
    # Get all .tif files from the folder
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    # Sort files to maintain any order if needed
    tif_files.sort()
    
    # Set up the plot - number of plots will be twice the number of .tif files (image + histogram)
    num_files = len(tif_files)
    fig, axs = plt.subplots(num_files, 2, figsize=(10, 5 * num_files))  # Adjust size as needed

    for i, file_name in enumerate(tif_files):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the .tif file
        with rasterio.open(file_path) as src:
            image = src.read(1)  # Read the first band

        # Plot the image
        ax_img = axs[i, 0] if num_files > 1 else axs[0]
        im = ax_img.imshow(image, cmap='viridis')
        ax_img.set_title(f'Image: {file_name}')
        ax_img.axis('off')
        fig.colorbar(im, ax=ax_img, orientation='vertical')

        # Plot the histogram
        ax_hist = axs[i, 1] if num_files > 1 else axs[1]
        ax_hist.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
        ax_hist.set_title(f'Histogram: {file_name}')
        ax_hist.set_xlabel('Pixel values')
        ax_hist.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


