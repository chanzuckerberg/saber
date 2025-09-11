"""
Shared utility functions for 2D and 3D annotation viewers
"""
import numpy as np
import cv2
from typing import Optional, Dict, Any
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


def get_boundary_opencv_fast(mask: np.ndarray) -> Optional[np.ndarray]:
    """Fast boundary detection using OpenCV with aggressive optimization."""
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if largest.shape[1] == 1:
        pts = largest.squeeze(axis=1)
    else:
        pts = largest.reshape(-1, 2)
    # Subsample
    if len(pts) > 100:
        step = max(1, len(pts) // 50)
        pts = pts[::step]
    return pts


def create_overlay_rgba(mask: np.ndarray, index: int = 0, class_dict: Dict = None, class_name: str = None) -> np.ndarray:
    """Create colored overlay for mask using TAB10 colors"""
    TAB10_COLORS = [
        (31/255, 119/255, 180/255),   # blue
        (255/255, 127/255, 14/255),   # orange
        (44/255, 160/255, 44/255),    # green
        (214/255, 39/255, 40/255),    # red
        (148/255, 103/255, 189/255),  # purple
        (140/255, 86/255, 75/255),    # brown
        (227/255, 119/255, 194/255),  # pink
        (0/255, 128/255, 128/255),    # teal
        (188/255, 189/255, 34/255),   # olive
        (23/255, 190/255, 207/255),   # cyan
    ]
    
    Nx, Ny = mask.shape
    rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)
    
    if class_name is not None and class_dict is not None and class_name in class_dict:
        index = class_dict[class_name]['value'] - 1
    
    color = TAB10_COLORS[index % len(TAB10_COLORS)]
    
    inds = mask > 0.5
    rgba[inds, 0] = color[0]
    rgba[inds, 1] = color[1]
    rgba[inds, 2] = color[2]
    rgba[inds, 3] = 1.0
    
    return rgba


def extract_masks_from_labels(labels: np.ndarray) -> tuple:
    """
    Extract individual masks from a 2D label map
    Returns: (masks_list, mask_values)
    """
    mask_values = np.unique(labels[labels > 0])
    masks = []
    for val in mask_values:
        masks.append((labels == val).astype(np.float32))
    return masks, mask_values


def process_mask_data(masks: np.ndarray) -> tuple:
    """
    Process mask data whether it's a 2D label map or 3D stack of masks
    Returns: (extracted_masks, mask_values)
    """
    if len(masks.shape) == 2:
        # It's a 2D label map - extract individual masks
        return extract_masks_from_labels(masks)
    elif len(masks.shape) == 3:
        # It's already a stack of masks - extract their values
        mask_values = []
        for i, mask in enumerate(masks):
            unique_vals = np.unique(mask[mask > 0])
            if len(unique_vals) > 0:
                mask_values.append(unique_vals[0])
            else:
                mask_values.append(i + 1)
        return masks, mask_values
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")


def create_mask_overlay_item(mask: np.ndarray, index: int, class_dict: Dict = None, 
                            class_name: str = None, opacity: float = 0.4) -> pg.ImageItem:
    """Create a pyqtgraph ImageItem for a mask overlay"""
    overlay = create_overlay_rgba(mask, index, class_dict, class_name)
    item = pg.ImageItem(overlay)
    item.setOpacity(opacity)
    item.setZValue(index + 1)
    return item


def create_boundary_item(mask: np.ndarray, index: int, 
                        pen_color: str = 'w', pen_width: int = 2) -> pg.PlotDataItem:
    """Create a pyqtgraph PlotDataItem for mask boundary"""
    boundary = pg.PlotDataItem(pen=pg.mkPen(color=pen_color, width=pen_width, 
                                           style=QtCore.Qt.SolidLine))
    boundary.setZValue(1000 + index)
    boundary.setVisible(False)
    
    # Get and set boundary points
    boundary_pts = get_boundary_opencv_fast(mask)
    if boundary_pts is not None and len(boundary_pts) > 0:
        boundary_pts = np.vstack([boundary_pts, boundary_pts[0:1]])
        boundary.setData(boundary_pts[:, 1], boundary_pts[:, 0])
    
    return boundary


def highlight_mask_with_boundary(mask_idx: int, left_mask_items: list, right_mask_items: list,
                                left_boundary_items: list, right_boundary_items: list):
    """Highlight a specific mask with boundary on the appropriate panel"""
    # Clear all boundaries first
    for boundary in left_boundary_items:
        boundary.setVisible(False)
    for boundary in right_boundary_items:
        boundary.setVisible(False)
    
    # Show boundary on appropriate panel
    if mask_idx < len(right_mask_items) and right_mask_items[mask_idx].isVisible():
        right_boundary_items[mask_idx].setVisible(True)
    elif mask_idx < len(left_mask_items) and left_mask_items[mask_idx].isVisible():
        left_boundary_items[mask_idx].setVisible(True)


def clear_all_highlights(left_boundary_items: list, right_boundary_items: list):
    """Clear all boundary highlights"""
    for boundary in left_boundary_items:
        boundary.setVisible(False)
    for boundary in right_boundary_items:
        boundary.setVisible(False)