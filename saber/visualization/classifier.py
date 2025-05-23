from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import cv2

def add_masks(masks, ax):

    # Define unique colors for each class (RGBA values)
    colors = [
        (1, 0, 0, 0.5),  # Red with transparency
        (0, 1, 0, 0.5),  # Green with transparency
        (0, 0, 1, 0.5),  # Blue with transparency
        (1, 1, 0, 0.5),  # Yellow with transparency
        (0.5, 0, 0.5, 0.5),  # Purple with transparency
        (1, 0.5, 0, 0.5),  # Orange with transparency
        (0, 1, 1, 0.5),  # Cyan with transparency
        (1, 0, 1, 0.5),  # Magenta with transparency
    ]

    num_masks = masks.shape[0]
    for i in range(num_masks):
        
        # Cycle through colors if there are more masks than colors
        color = colors[i % len(colors)]  

        # Create a custom colormap for this mask
        custom_cmap = ListedColormap([
            (1, 1, 1, 0),  # Transparent white for 0 values
            color,  # Assigned color for non-zero values
        ])

        ax.imshow(masks[i], cmap=custom_cmap, alpha=0.6)
    ax.axis('off')
    # plt.tight_layout()
    # plt.show()    


def display_masks(im, masks, masks2=None, title=None):
    """
    Display a grayscale image with overlaid masks in different colors.
    
    Args:
        im (numpy.ndarray): The grayscale image to display.           [H, W]
        masks (numpy.ndarray): The masks to overlay on the image.  [N, H, W]
    """

    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].imshow(im, cmap='gray'); ax[0].axis('off')
    ax[1].imshow(im, cmap='gray'); ax[1].axis('off')
    if masks2 is not None:
        add_masks(masks2, ax[0])
    add_masks(masks, ax[1])
    plt.tight_layout()    

    # Add a centered title above both images
    if title is not None: fig.suptitle(title, fontsize=16, y=1.03)

def plot_metrics(train_array, validation_array, metric_name="Metric", save_path=None):
    """
    Plots training and validation metrics over epochs.

    Parameters:
    - train_array (list or numpy array): Array of training metric values.
    - validation_array (list or numpy array): Array of validation metric values.
    - metric_name (str): Name of the metric to display on the plot.

    Returns:
    - None
    """
    epochs = np.arange(1, len(train_array) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_array[:], label="Training", marker='o', linestyle='-')
    plt.plot(epochs, validation_array[:], label="Validation", marker='s', linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()  

    if save_path: 
        plt.savefig(save_path)

def plot_all_metrics(metrics, save_path=None):
    """
    Plots multiple training and validation metrics on a single figure with subplots.

    Parameters:
    - metrics (dict): Dictionary with keys 'train' and 'val'. Each key maps to a dictionary 
                      of metric names to arrays/lists of metric values.
                      e.g., 
                      {
                          'train': {
                              'loss': [...],
                              'accuracy': [...],
                              ...
                          },
                          'val': {
                              'loss': [...],
                              'accuracy': [...],
                              ...
                          }
                      }
    - save_path (str): Path to save the figure. The file extension determines the format (e.g., .png or .pdf).

    Returns:
    - None
    """
    # Extract metric names (assuming both train and val have the same keys)
    metric_names = list(metrics['train'].keys())
    num_metrics = len(metric_names)
    
    # For simplicity, we'll arrange the plots vertically.
    n_rows = num_metrics
    n_cols = 1
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 9, n_rows * 2))
    axs = axs.flatten() if num_metrics > 1 else [axs]
    
    # Assume all metrics have the same number of epochs; use the first metric from training.
    first_metric = metric_names[0]
    epochs = np.arange(1, len(metrics['train'][first_metric]) + 1)
    
    for i, metric_name in enumerate(metric_names):
        train_data = metrics['train'][metric_name]
        val_data = metrics['val'][metric_name]
        ax = axs[i]
        
        ax.plot(epochs, train_data, label="Training", marker='o', linestyle='-')
        ax.plot(epochs, val_data, label="Validation", marker='s', linestyle='--')
        ax.set_ylabel(metric_name)
        ax.set_xlim(1, epochs[-1])
        ax.grid(True)
        
        if i == num_metrics - 1:
            ax.set_xlabel("Epochs")
            ax.legend()
        else:
            ax.set_xticklabels([])

    # Remove any extra subplots if they exist.
    for ax in axs[num_metrics:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_per_class_metrics(per_class_results, save_path=None):
    """
    Plots per-class metrics stored in a dictionary with structure:
    
        {
            'train': { 'class0': { 'precision': [...], 'recall': [...], 'f1_score': [...] },
                       'class1': { ... },
                       ... },
            'val':   { 'class0': { 'precision': [...], 'recall': [...], 'f1_score': [...] },
                       'class1': { ... },
                       ... }
        }
    
    The plot will be arranged in a 3x2 grid where:
      - Each row corresponds to one metric (precision, recall, f1_score).
      - The first column shows training curves and the second column shows validation curves.
    
    The background class ("class0") is skipped.
    Only the bottom row shows x-tick labels.
    """
    # Get the metric names from any (non-empty) class in train mode.
    # (We assume all classes have the same metric keys.)
    some_class = next(iter(per_class_results['train'].values()))
    metric_names = list(some_class.keys())
    num_metrics = len(metric_names)
    
    # Create a 3x2 grid (rows: metrics, cols: train and val)
    fig, axs = plt.subplots(num_metrics, 2, figsize=(12, num_metrics * 3))
    
    # Determine epochs from one non-background class in train mode.
    sample_list = None
    for cls_key, metrics in per_class_results['train'].items():
        if cls_key != "class0":
            sample_list = metrics[metric_names[0]]
            break
    if sample_list is None or len(sample_list) == 0:
        print("No non-background classes with data found.")
        return
    
    epochs = np.arange(1, len(sample_list) + 1)
    
    # Loop over each metric (row) and mode (column)
    for i, metric in enumerate(metric_names):
        for j, mode in enumerate(['train', 'val']):
            ax = axs[i, j]
            # Plot curves for all classes except "class0"
            for cls_key, metrics in per_class_results[mode].items():
                if cls_key == "class0":
                    continue
                ax.plot(
                    epochs,
                    metrics[metric],
                    label=cls_key,
                    marker='o',
                    linestyle='-'
                )
            # Only add x-tick labels for the bottom row.
            if i == num_metrics - 1:
                ax.set_xlabel("Epochs")
                ax.legend()
            else:
                ax.set_xticklabels([])
            if len(epochs) > 0:
                ax.set_xlim(1, epochs[-1])
            ax.set_ylim(0.4, 1)

            # Title only on first row and y label only on first column
            if i == 0:
                ax.set_title(f"{mode}")
            if j == 0:
                ax.set_ylabel(metric)                
            ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
