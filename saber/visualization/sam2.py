from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def display_video_segmentation(
    video_segments,
    inference_state, 
    frame_stride: int = 30
    ):

    # Create a figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(bottom=0.25)  # Adjust layout to leave space for the slider

    # Initial frame to display
    initial_frame = 0

    # Display the first frame
    show_tomo_frame(inference_state["images"], initial_frame, ax1)
    show_tomo_frame(inference_state["images"], initial_frame, ax2)

    for out_obj_id, out_mask in video_segments[initial_frame].items():
        show_mask1(out_mask, ax2, obj_id=out_obj_id)
    ax2.axis('off')

    # Create the slider axes and the slider widget
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # [left, bottom, width, height` `]
    frame_slider = Slider(ax_slider, 'Frame', 0, len(video_segments) - frame_stride, valinit=initial_frame, valstep=frame_stride)

    # Define the update function to change the frame
    def update(val):
        frame_idx = int(frame_slider.val)

        # Clear both axes
        ax1.clear()
        ax2.clear()

        # Update the plot with the new frame
        ax1.set_title(f"frame {frame_idx}")
        show_tomo_frame(inference_state["images"], frame_idx, ax1)
        show_tomo_frame(inference_state["images"], frame_idx, ax2)

        # Overlay the masks
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            show_mask1(out_mask, ax2, obj_id=out_obj_id)

        ax2.axis('off')
        
        # Redraw the updated plot
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    frame_slider.on_changed(update)

    # Display the figure with the slider
    plt.show()


##################### Meta FAIR Utility Functions #####################

def show_mask1(mask, ax, obj_id=None, random_color=False):
    """
    Overlay a binary mask onto an image with a unique color.

    Args:
        mask (numpy.ndarray): A binary mask with shape [H, W].
        ax (matplotlib.axes.Axes): The axis to plot the mask on.
        obj_id (int, optional): Identifier for the object mask, used for consistent coloring. Defaults to None.
        random_color (bool, optional): If True, assigns a random color to the mask. Defaults to False.
    """    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.7])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """
    Display positive and negative points on an image.

    Args:
        coords (numpy.ndarray): Array of point coordinates with shape [N, 2].
        labels (numpy.ndarray): Binary labels for each point, where 1 = positive (green) and 0 = negative (red).
        ax (matplotlib.axes.Axes): The axis to plot the points on.
        marker_size (int, optional): Size of the markers. Defaults to 375.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size*1.5, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    """
    Draw a bounding box on an image.

    Args:
        box (list or numpy.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        ax (matplotlib.axes.Axes): The axis to plot the box on.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Display multiple masks on an image, along with optional points and bounding boxes.

    Args:
        image (numpy.ndarray): The image to display, shape [H, W, C].
        masks (list of numpy.ndarray): List of binary masks to overlay on the image.
        scores (list of float): Confidence scores for each mask.
        point_coords (numpy.ndarray, optional): Coordinates of key points to overlay. Defaults to None.
        box_coords (list, optional): Bounding box coordinates to overlay. Defaults to None.
        input_labels (numpy.ndarray, optional): Labels for points. Defaults to None.
        borders (bool, optional): Whether to draw mask borders. Defaults to True.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask2(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_anns(anns, borders=True):
    """
    Display multiple segmented annotations on an image.

    Args:
        anns (list of dict): List of annotation dictionaries with key 'segmentation' (numpy.ndarray) and 'area'.
        borders (bool, optional): Whether to draw the contours of masks. Defaults to True.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def show_tomo_frame(tomo, frame_id, ax):
    frame = torch.tensor(tomo[frame_id], device="cpu")
    frame -= frame.min()
    frame /= frame.max()
    if frame.ndim == 3:
        frame = frame[0]

    ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Frame {frame_id}")

def show_mask2(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def plot_frame_scores(data, func, a_fit, b_fit, c_fit):

    # Create x values (indices)
    x = np.arange(len(data))

    # Calculate R^2 value
    residuals = data - func(x, a_fit, b_fit, c_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot the Fit (Debug)
    plt.plot(data, label='Original data')
    # Plot the fitted curve
    plt.plot(func(x, a_fit, b_fit, c_fit), 'r--', 
            label=f'Fit: {a_fit:.2e}*(x-{b_fit:.1f})²+{c_fit:.1f}, R²={r_squared:.3f}')

    plt.xlim([0, len(data)]); 
    plt.xlabel('Slice Along Z-axis'); 
    plt.ylabel('Object Score Logits')
    plt.grid(True)
    plt.tick_params(direction='in', top=True, right=True, length=6, width=1)
    plt.legend()
    plt.show()

def plot_fit(data, func, fit_params):
    """
    Plot the Regression of Confidence Scores Along Z-Axis
    """

    # Create x values (indices)
    x = np.arange(len(data))

    # Plot the Original Data
    plt.plot(data, label='Original data')

    # Plot the Fitted Curve
    plt.plot(func(x, *fit_params), 'r--', label='Fitted curve')

    # Label the Axes
    plt.xlabel('Slice Along Z-axis')
    plt.ylabel('Object Score Logits')
    
    plt.grid(True)
    plt.tick_params(direction='in', top=True, right=True, length=6, width=1)
    plt.legend()
    plt.show()