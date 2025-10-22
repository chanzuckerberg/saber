from saber.visualization import classifier, sam2 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio, os
import numpy as np

def save_slab_segmentation(
    current_run,
    image, masks,
    show_plot: bool = False
    ):

    # Show 2D Annotations
    plt.imshow(image, cmap='gray'); plt.axis('off')
    if len(masks) > 0: # I Should Update this Function as Well...
         classifier.display_mask_list(image, masks)
    plt.axis('off')

    # Save the Figure
    runID, sessionID = current_run.split('-')
    os.makedirs(f'gallery_sessionID_{sessionID}/frames', exist_ok=True)
    plt.savefig(f'gallery_sessionID_{sessionID}/frames/{runID}.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

def export_movie(vol, vol_masks, output_path='segmentation_movie.gif', fps=5):
    """
    Create a movie using imageio (supports GIF and MP4).

    Args:
        vol: 3D array of images (frames, height, width)
        vol_masks: 3D array of masks (frames, height, width)
        output_path: Path to save the movie (.gif or .mp4)
        fps: Frames per second
    """

    def _masks_to_array(masks):
        """Helper function if you need it"""
        if isinstance(masks, list):
            return np.array(masks)
        return masks

    # Get colors
    colors = classifier.get_colors()
    max_mask_value = np.max(vol_masks)
    cmap_colors = [(1, 1, 1, 0)] + colors[:max_mask_value]  # 0 is transparent
    cmap = ListedColormap(cmap_colors)

    print(f"Processing {len(vol)} frames...")
    frames = []
    for i in tqdm(range(len(vol))):
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')

        # Plot image
        ax.imshow(vol[i], cmap='gray')

        # Plot masks
        masks = vol_masks[i]
        if isinstance(masks, list):
            masks = _masks_to_array(masks)
        ax.imshow(masks, cmap=cmap, alpha=0.6, vmin=0, vmax=max_mask_value)

        # Add frame number
        ax.text(0.02, 0.95, f'Frame: {i + 1}/{len(vol)}',
                transform=ax.transAxes, fontsize=16, color='white', weight='bold')

        # Convert plot to image array (matplotlib compatibility fix)
        fig.canvas.draw()
        try:
            # Try newer matplotlib method first
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3]  # Remove alpha channel
        except AttributeError:
            # Fallback for older matplotlib versions
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(frame)
        plt.close(fig)

    # Save as movie
    print(f"Saving {len(frames)} frames to {output_path}...")

    if output_path.endswith('.gif'):
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:  # MP4 or other video format
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')

    print("Movie saved successfully!")
    return frames

