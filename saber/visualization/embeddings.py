import matplotlib.pyplot as plt
from umap import UMAP
import numpy as np
import colorsys

def visualize_patch_features(features, num_channels=16):
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


def visualize_patch_features_umap(features, n_neighbors=15, min_dist=0.1):
    """
    Project patch embeddings to 2D using UMAP, then map the 2D coordinates to color.

    Parameters
    ----------
    features : np.ndarray
        Patch features with shape (C, H, W), where:
            C = embedding dimension (e.g. 768 for DinoV2),
            H = number of patches vertically,
            W = number of patches horizontally.
    n_neighbors : int
        UMAP parameter controlling local neighborhood size.
    min_dist : float
        UMAP parameter controlling how tightly points can be packed.

    Returns
    -------
    color_image : np.ndarray
        A (H, W, 3) RGB image visualizing the patch features via UMAP.
    """
    # 1) Flatten from (C, H, W) -> (H*W, C)
    C, H, W = features.shape
    patch_vectors = features.reshape(C, -1).T  # shape = (H*W, C)

    # 2) Run UMAP to reduce to 2D
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42  # for reproducibility
    )
    embedding_2d = reducer.fit_transform(patch_vectors)  # shape = (H*W, 2)

    # 3) Normalize the 2D embedding to [0, 1] range
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()

    # Avoid division by zero by adding a small epsilon
    embedding_2d[:, 0] = (embedding_2d[:, 0] - x_min) / (x_max - x_min + 1e-8)
    embedding_2d[:, 1] = (embedding_2d[:, 1] - y_min) / (y_max - y_min + 1e-8)

    # 4) Map each 2D point to a color.
    #    For instance, treat (u, v) as (hue, saturation) in HSV space with value=1.
    rgb_colors = []
    for (u, v) in embedding_2d:
        h = u       # hue
        # Scale saturation and value
        s = 0.95 * v    # 70% of original saturation
        val = 0.7     # 80% brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, val)
        rgb_colors.append([r, g, b])

    rgb_colors = np.array(rgb_colors).reshape(H, W, 3)

    return rgb_colors
