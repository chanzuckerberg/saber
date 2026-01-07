import matplotlib.pyplot as plt
import numpy as np
import colorsys

def patch_features(features, num_channels=16):
    """
    Visualize patch features with a distinct color per channel.

    Parameters:
    -----------
    features : numpy.ndarray
        A numpy array of shape (1, channels, height, width) or (channels, height, width)
        containing patch features.
    num_channels : int
        The number of channels to visualize.

    Returns:
    --------
    composite : numpy.ndarray
        A composite RGB image showing the patch features color-coded by channel.
    """
    # If there's a batch dimension, remove it.
    if features.ndim == 4:
        features = features[0]  # Now shape is (C, H, W)

    channels, H, W = features.shape

    # Normalize each channel independently to the [0, 1] range.
    norm_features = np.zeros_like(features)
    for c in range(num_channels):
        channel = features[c]
        min_val, max_val = channel.min(), channel.max()
        norm_features[c] = (channel - min_val) / (max_val - min_val + 1e-8)

    # Generate a distinct color for each channel using an HSV colormap.
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / num_channels) for i in range(num_channels)]  # each color is an RGBA tuple

    # Create a composite image.
    composite = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(num_channels):
        # Get the RGB part of the color (ignore alpha).
        color = np.array(colors[c][:3])
        # Multiply the normalized channel with its assigned color and add it to the composite.
        composite += norm_features[c, :, :][..., np.newaxis] * color

    # Normalize the composite image to the [0, 1] range.
    composite -= composite.min()
    composite /= (composite.max() + 1e-8)

    return composite