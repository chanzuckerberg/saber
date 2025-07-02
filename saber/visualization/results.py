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
        sam2.show_anns(masks) 
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
        imageio.mimsave(output_path, frames, fps=fps)
    else:  # MP4 or other video format
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264')

    print("Movie saved successfully!")
    return frames

# def record_video_segmentation(video_segments, 
#                               inference_state, 
#                               frame_stride: int = 5, 
#                               output_file="segmentation_output.mp4", 
#                               fps=10):
#     from IPython.display import Video
#     dpi = 300

#     # Normalize the images to be between 0 and 1
#     inference_state['images'] = inference_state['images'] - inference_state['images'].min()
#     inference_state['images'] = inference_state['images'] / inference_state['images'].max()

#     # Video settings
#     _, frame_height, frame_width = inference_state["images"][0].shape  # Ensure RGB images (height, width, channels)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
#     out_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width * 2, frame_height))  # *2 because of 2 side-by-side frames

#     # Process each frame with the given frame_stride
#     for frame_idx in range(0, len(video_segments), frame_stride):

#         frame_image = create_segmentation_frame(
#                         frame_idx,
#                         frame_width,
#                         frame_height,
#                         dpi,
#                         inference_state, 
#                         video_segments )

#         # Write the combined frame to the video file - Convert RGB to BGR for OpenCV
#         out_video.write(cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR))  

#     # Process each frame with the given frame_stride - Reverse Order Now
#     for frame_idx in range(len(video_segments) - 1, -1, -frame_stride):

#         frame_image = create_segmentation_frame(
#                         frame_idx,
#                         frame_width,
#                         frame_height,
#                         dpi,
#                         inference_state, 
#                         video_segments )

#         # Write the combined frame to the video file - Convert RGB to BGR for OpenCV
#         out_video.write(cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR))          

#     # Release the video writer object
#     out_video.release()

#     print(f"Video saved as {output_file}")

#     # Display the video in Jupyter Notebook
#     return Video(output_file, embed=True)

# def create_segmentation_frame(
#     frame_idx,
#     frame_width,
#     frame_height,
#     dpi,
#     inference_state, 
#     video_segments,
#     ):

#     # Recreate figure and axes for each frame (fixing the size)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(frame_width * 2 / dpi, frame_height / dpi), dpi=dpi)

#     # Display the original frame on ax1
#     viz.show_tomo_frame(inference_state["images"], frame_idx, ax1)
#     ax1.set_title(f"Frame {frame_idx}")
#     ax1.axis('off')

#     # Display the frame with segmentation masks on ax2
#     viz.show_tomo_frame(inference_state["images"], frame_idx, ax2)
#     for out_obj_id, out_mask in video_segments[frame_idx].items():
#         viz.show_mask1(out_mask, ax2, obj_id=out_obj_id)
#     ax2.set_title(f"Frame {frame_idx} with Segmentation")
#     ax2.axis('off')

#     # Convert the figure to an image array
#     fig.canvas.draw()  # Ensure the canvas is drawn

#     # Get the canvas width and height
#     width, height = fig.canvas.get_width_height()
#     # print("Canvas size:", width, height)

#     # # Convert the canvas to a NumPy array and reshape it to (height, width, 3)
#     # frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     # if frame_image.size != height * width * 3:
#     #     raise ValueError(f"Unexpected frame size: expected {height * width * 3}, got {frame_image.size}")
#     # frame_image = frame_image.reshape((height, width, 3))

#     # Get ARGB buffer
#     buffer = fig.canvas.tostring_argb()
#     # Convert to a NumPy array and reshape it to (height, width, 4)
#     frame_image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
#     # Convert from ARGB to RGB by dropping the alpha channel
#     frame_image = frame_image[..., 1:]

#     # Close the figure to free up memory
#     plt.close(fig)

#     # Ensure that frame_image and combined_frame are the same size before assignment
#     if frame_image.shape[0] != frame_height or frame_image.shape[1] != frame_width * 2:
#         raise ValueError(f"Frame size mismatch: expected ({frame_height}, {frame_width * 2}), but got {frame_image.shape[:2]}")

#     return frame_image