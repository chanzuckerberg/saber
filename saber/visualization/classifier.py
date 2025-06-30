from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import cv2

def display_mask_list(image: np.ndarray, masks: list):
    """
    Display a list of masks in a single image.
    """

    # If Masks is an Array, Display it
    if isinstance(masks, np.ndarray):
        display_mask_array(image, masks)

    # If Masks is an Empty List, Display the Image
    if len(masks) == 0:
        print("No masks found")
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.axis('off'); plt.show()
    else:
        # Sort Masks so that smallest masks are on top. 
        masks = sorted(masks, key=lambda mask: mask['area'], reverse=True)

        # Convert Masks to Array if List
        masks = _masks_to_array(masks)

        # Display the Masks
        display_mask_array(image, masks)

def display_mask_array(image: np.ndarray, masks: np.ndarray):

    # Get colors
    colors = get_colors()

    # Generate Figure and Plot Image
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')

    # Plot the Segmentations over the Image
    cmap_colors = [(1, 1, 1, 0)] + colors[:np.max(masks)]  # 0 is transparent
    cmap = ListedColormap(cmap_colors)
    plt.imshow(masks, cmap=cmap, alpha=0.6)
    plt.axis('off'); plt.show()

def _masks_to_array(masks):
    """
    Convert list of masks to single label matrix
    
    Args:
        masks: List of mask dictionaries with 'segmentation' key
        image_shape: Shape of the output matrix (height, width)
    
    Returns:
        label_matrix: numpy array where each mask has unique ID (1 to N)
    """
    
    # Return a (Nx, Ny) matrix where each pixel is labeled with the mask ID
    label_matrix = np.zeros(masks[0]['segmentation'].shape, dtype=np.uint16)
    for idx, mask in enumerate(masks, start=1):
        label_matrix[mask['segmentation'] > 0] = idx
    
    return label_matrix

def masks_to_3d_array(masks):

    classes = np.unique(masks)
    max_val = classes.max()
    if max_val == 1:
        return np.expand_dims(masks, axis=0) # Shape: (1, H, W)
    else:
        # Create a one-hot encoded array
        one_hot = np.zeros((max_val + 1, *masks.shape), dtype=np.uint8)
        for cls in np.arange(1, max_val + 1):
            one_hot[cls-1] = (masks == cls).astype(np.uint8)
        return one_hot

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

def get_colors():

    # Extended vibrant color palette
    colors = [
        (0, 1, 1, 0.5),        # Cyan (bright, high contrast)
        (1, 0, 1, 0.5),        # Magenta
        (0, 0, 1, 0.5),        # Blue
        (0, 1, 0, 0.5),        # Green

        (1, 0.5, 0, 0.5),      # Orange
        (0.5, 0, 0.5, 0.5),    # Purple
        (0.2, 0.6, 0.9, 0.5),  # Sky Blue
        (0.9, 0.2, 0.6, 0.5),  # Hot Pink
        (0.6, 0.2, 0.8, 0.5),  # Violet

        (0.4, 0.7, 0.2, 0.5),  # Lime
        (0.8, 0.4, 0, 0.5),    # Burnt Orange
        (0, 0.5, 0, 0.5),      # Dark Green
        (0.7, 0.3, 0.6, 0.5),  # Orchid
        (0.9, 0.6, 0.2, 0.5),  # Gold

        (1, 1, 0.3, 0.5),      # Yellow
        (0.5, 0.5, 0, 0.5),    # Olive
        (0, 0, 0.5, 0.5),      # Navy
        (0.5, 0, 0, 0.5),      # Maroon

        # Pastel shades (can be used for less prominent classes)
        (1, 0.7, 0.7, 0.5),    # Light Red/Pink
        (0.7, 1, 0.7, 0.5),    # Light Green
        (0.7, 0.7, 1, 0.5),    # Light Blue
        (1, 1, 0.7, 0.5),      # Light Yellow
    ]

    return colors

def add_masks(masks, ax):

    # Get colors
    colors = get_colors()

    # Get number of masks
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


# def _display_mask_list(masks, ax):
#     """
#     Display a list of masks in a single image.
#     """
#     # Get colors
#     colors = get_colors()

#     # Get number of masks
#     num_masks = masks.shape[0]
#     for i in range(num_masks):
        
#         # Cycle through colors if there are more masks than colors
#         color = colors[i % len(colors)]  

#         # Create a custom colormap for this mask
#         custom_cmap = ListedColormap([
#             (1, 1, 1, 0),  # Transparent white for 0 values
#             color,  # Assigned color for non-zero values
#         ])

#         ax.imshow(masks[i], cmap=custom_cmap, alpha=0.6)
#     ax.axis('off')
#     # plt.tight_layout()
#     # plt.show() 