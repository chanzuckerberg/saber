import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available, using slower boundary detection")

from typing import Dict, Optional
from saber.gui.base.segmentation_picker import SegmentationViewer

class HashtagSegmentationViewer(SegmentationViewer):
    """Enhanced SegmentationViewer with hashtag-based coloring and selection highlighting."""
    
    def __init__(self, image, masks):
        super().__init__(image, masks)
        
        # Track selection 
        self.selected_mask_id = None
        self.selection_boundary_item = None
        
        # Store custom colors for masks (mask_id -> (r, g, b))
        self.custom_colors = {}
        
        # Fast boundary cache to avoid recomputing
        self.boundary_cache = {}
        
    def update_mask_colors(self, color_mapping: Dict[int, str]):
        """Update mask overlay colors based on hashtag colors.
        
        Args:
            color_mapping: dict mapping mask_id -> hex_color_string (e.g., '#FF6B6B')
        """
        # Check if colors have actually changed
        new_custom_colors = {}
        
        for mask_id, hex_color in color_mapping.items():
            if hex_color.startswith('#'):
                hex_color = hex_color[1:]
            
            try:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0  
                b = int(hex_color[4:6], 16) / 255.0
                new_custom_colors[mask_id] = (r, g, b)
            except (ValueError, IndexError):
                print(f"Invalid hex color: {hex_color}")
                continue
        
        # Only update if colors actually changed
        if new_custom_colors != self.custom_colors:
            self.custom_colors = new_custom_colors
            self.refresh_overlays()
    
    def create_overlay_rgba(self, mask, index=0):
        """
        Override to use custom colors when available, otherwise fall back to tab10.
        """
        Nx, Ny = mask.shape
        rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)

        # Use custom color if available, otherwise use tab10
        if index in self.custom_colors:
            color = self.custom_colors[index]
        else:
            color = self.tab10_colors[index % len(self.tab10_colors)]

        # Apply the color to the mask
        inds = mask > 0.5
        rgba[inds, 0] = color[0]  # Red channel
        rgba[inds, 1] = color[1]  # Green channel
        rgba[inds, 2] = color[2]  # Blue channel
        rgba[inds, 3] = 1.0       # Alpha channel
        
        return rgba
    
    def refresh_overlays(self):
        """Refresh only visible mask overlays with current colors."""
        for i, mask in enumerate(self.masks):
            if i < len(self.left_mask_items) and self.left_mask_items[i].isVisible():
                new_rgba = self.create_overlay_rgba(mask, i)
                self.left_mask_items[i].setImage(new_rgba)
                
            if (i < len(self.right_mask_items) and 
                hasattr(self.right_mask_items[i], 'isVisible') and 
                self.right_mask_items[i].isVisible()):
                new_rgba = self.create_overlay_rgba(mask, i)
                self.right_mask_items[i].setImage(new_rgba)
    
    def highlight_mask(self, mask_id: int):
        """Add a boundary highlight around a specific mask and update selected_mask_id."""
        if mask_id >= len(self.masks) or mask_id < 0:
            return
        
        # Remove existing boundary WITHOUT clearing selected_mask_id
        if self.selection_boundary_item is not None:
            try:
                self.right_view.removeItem(self.selection_boundary_item)
            except:
                try:
                    self.left_view.removeItem(self.selection_boundary_item)
                except:
                    pass  # Already removed or never added
            self.selection_boundary_item = None
        
        # NOW set the selected mask ID (after clearing the old boundary but before creating new one)
        self.selected_mask_id = mask_id
        
        # Only highlight if the mask is visible on the right panel (i.e., accepted)
        if (hasattr(self, 'right_mask_items') and 
            mask_id < len(self.right_mask_items) and 
            self.right_mask_items[mask_id].isVisible()):
            
            # Get boundary points
            boundary_points = self.get_mask_boundary(mask_id)
            
            # Check if boundary_points is None or empty array
            if boundary_points is None or len(boundary_points) == 0:
                return
            
            # Create boundary item for the RIGHT panel
            self.selection_boundary_item = pg.PlotDataItem(
                boundary_points[:, 1], boundary_points[:, 0],  # x, y coordinates
                pen=pg.mkPen(color='white', width=3, style=QtCore.Qt.DashLine),
                connect='finite'
            )
            # Add to RIGHT view where the accepted segmentations are shown
            self.right_view.addItem(self.selection_boundary_item)

    def clear_highlight(self):
        """Clear the selection highlight and reset selected_mask_id."""
        if self.selection_boundary_item is not None:
            # Remove from whichever view it's in
            try:
                self.right_view.removeItem(self.selection_boundary_item)
            except:
                try:
                    self.left_view.removeItem(self.selection_boundary_item)
                except:
                    pass  # Already removed or never added
            self.selection_boundary_item = None
        # Reset selected mask ID when clearing highlight
        self.selected_mask_id = None
    
    def get_mask_boundary(self, mask_id: int) -> Optional[np.ndarray]:
        """Get boundary points for a mask using fast method with caching."""
        # Check cache first with hash-based key for better cache efficiency
        cache_key = f"{mask_id}_{hash(str(self.masks[mask_id].tobytes()))}"
        if hasattr(self, '_boundary_cache_keys') and cache_key in self._boundary_cache_keys:
            return self.boundary_cache.get(mask_id)
        
        mask = self.masks[mask_id]
        
        if HAS_OPENCV:
            boundary_points = self._get_boundary_opencv_fast(mask)
        else:
            boundary_points = self._get_boundary_numpy_fast(mask)
        
        # Cache with size limit to prevent memory issues
        if not hasattr(self, '_boundary_cache_keys'):
            self._boundary_cache_keys = set()
            
        # Limit cache size to prevent memory bloat
        if len(self.boundary_cache) > 50:
            # Clear half the cache
            keys_to_remove = list(self.boundary_cache.keys())[:25]
            for key in keys_to_remove:
                del self.boundary_cache[key]
            self._boundary_cache_keys = set(self.boundary_cache.keys())
        
        self.boundary_cache[mask_id] = boundary_points
        self._boundary_cache_keys.add(cache_key)
        return boundary_points
    
    def _get_boundary_opencv_fast(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Fast boundary detection using OpenCV with aggressive optimization."""
        try:
            # Convert to uint8 and ensure binary
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            
            # Use faster contour approximation
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            
            if not contours:
                return None
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            if largest_contour.shape[1] == 1:
                boundary_points = largest_contour.squeeze(axis=1)
            else:
                boundary_points = largest_contour.reshape(-1, 2)
            
            # Aggressive subsampling for performance
            if len(boundary_points) > 100:
                step = len(boundary_points) // 50  # Limit to ~50 points max
                boundary_points = boundary_points[::step]
            
            return boundary_points
            
        except Exception as e:
            print(f"OpenCV boundary detection failed: {e}")
            return self._get_boundary_numpy_fast(mask)

    def load_data_fresh(self, base_image, masks):
        """Fresh data loading - clean slate approach."""
        # 1. Clear ALL state immediately
        self.boundary_cache.clear()
        if hasattr(self, '_boundary_cache_keys'):
            self._boundary_cache_keys.clear()
        self.custom_colors.clear()  
        self.clear_highlight()
        self.selected_mask_id = None
        
        # 2. Set new data
        self.image = base_image
        self.masks = masks
        
        # 3. Reset accepted masks to empty state
        if hasattr(self, 'accepted_masks'):
            self.accepted_masks.clear()
        else:
            self.accepted_masks = set()
            
        if hasattr(self, 'accepted_stack'):
            self.accepted_stack.clear()
        else:
            self.accepted_stack = []
        
        # 4. Update base images
        self.left_base_img_item.setImage(base_image)
        self.right_base_img_item.setImage(base_image)
        
        # 5. Remove ALL old overlays efficiently
        for item in getattr(self, 'left_mask_items', []):
            try:
                self.left_view.removeItem(item)
            except:
                pass
                
        for item in getattr(self, 'right_mask_items', []):
            try:
                self.right_view.removeItem(item)
            except:
                pass
        
        # 6. Create fresh overlay lists
        self.left_mask_items = []
        self.right_mask_items = []
        
        # 7. Initialize new overlays
        self.initialize_overlays()

    def load_data(self, base_image, masks, class_dict=None):
        """Override to clear caches when loading new data."""
        # Clear caches efficiently
        self.boundary_cache.clear()
        if hasattr(self, '_boundary_cache_keys'):
            self._boundary_cache_keys.clear()
        self.custom_colors.clear()
        self.clear_highlight()
        
        # Call parent method
        super().load_data(base_image, masks, class_dict)

    def mouse_clicked(self, event):
        """Override to handle clicks on both left and right panels."""
        # Get click position in scene coordinates
        scene_pos = event.scenePos()
        
        # Check if click is on left or right panel
        left_image_pos = self.left_base_img_item.mapFromScene(scene_pos)
        right_image_pos = self.right_base_img_item.mapFromScene(scene_pos)
        
        # Check bounds for left panel
        Nx, Ny = self.image.shape[:2]
        left_x, left_y = int(left_image_pos.x()), int(left_image_pos.y())
        right_x, right_y = int(right_image_pos.x()), int(right_image_pos.y())
        
        clicked_on_right = False
        clicked_on_left = False
        
        # Check if click is within right panel bounds
        if 0 <= right_x < Nx and 0 <= right_y < Ny:
            # Check if we clicked on an accepted mask in the right panel
            for i in range(len(self.masks)):
                if (hasattr(self, 'right_mask_items') and 
                    i < len(self.right_mask_items) and 
                    self.right_mask_items[i].isVisible() and
                    self.masks[i][right_x, right_y] > 0):
                    
                    # Clicked on an accepted segmentation in right panel
                    self.highlight_mask(i)
                    self.signal_segmentation_selected(i)
                    clicked_on_right = True
                    break
        
        # Check if click is within left panel bounds
        if not clicked_on_right and 0 <= left_x < Nx and 0 <= left_y < Ny:
            clicked_on_left = True
            # Use the original parent behavior for left panel clicks
            super().mouse_clicked(event)
            
            # After parent processes the click, check if mask was accepted and highlight
            for i in range(len(self.masks)):
                if (self.masks[i][left_x, left_y] > 0 and
                    hasattr(self, 'right_mask_items') and 
                    i < len(self.right_mask_items) and 
                    self.right_mask_items[i].isVisible()):
                    # Mask was just accepted, highlight it and signal selection
                    self.highlight_mask(i)
                    self.signal_segmentation_selected(i)
                    break
        
        # Clear highlight if clicked elsewhere
        if not clicked_on_right and not clicked_on_left:
            self.clear_highlight()
            self.signal_segmentation_deselected()

    def keyPressEvent(self, event):
            """
            Override to revert the currently selected mask instead of the last accepted mask.
            Press 'r' => revert the currently selected/highlighted mask
            """
            key = event.key()
            if key == QtCore.Qt.Key_R:
                # Check if we have a currently selected mask
                print(f"Selected mask ID: {self.selected_mask_id}")
                if self.selected_mask_id is not None:
                    mask_id = self.selected_mask_id
                    
                    # Only revert if the mask is currently accepted (visible on right panel)
                    if (mask_id in self.accepted_masks and
                        mask_id < len(self.right_mask_items) and 
                        self.right_mask_items[mask_id].isVisible()):
                        
                        # Remove from accepted set
                        self.accepted_masks.remove(mask_id)
                        
                        # Remove from accepted stack if present
                        if mask_id in self.accepted_stack:
                            self.accepted_stack.remove(mask_id)
                        
                        # Hide it on the right, show on the left
                        self.right_mask_items[mask_id].setVisible(False)
                        self.left_mask_items[mask_id].setVisible(True)
                        
                        # Clear the highlight since the mask is no longer accepted
                        self.clear_highlight()
                        
                        # Signal deselection to update the UI
                        self.signal_segmentation_deselected()
                        
                        print(f"Reverted selected mask {mask_id}: hidden on right, shown on left.")
                        print(f"Current accepted masks: {self.accepted_masks}")
                    else:
                        print(f"Selected mask {mask_id} is not currently accepted, cannot revert.")

    def signal_segmentation_selected(self, mask_id: int):
        """Signal that a segmentation was selected (for main window to handle)."""
        self.last_selected_mask_id = mask_id
        
        # If there's a callback function set, call it
        if hasattr(self, 'selection_callback') and self.selection_callback:
            self.selection_callback(mask_id)

    def signal_segmentation_deselected(self):
        """Signal that no segmentation is selected."""
        self.last_selected_mask_id = None
        if hasattr(self, 'deselection_callback') and self.deselection_callback:
            self.deselection_callback()

    def set_selection_callbacks(self, selection_callback=None, deselection_callback=None):
        """Set callback functions for when segmentations are selected/deselected."""
        self.selection_callback = selection_callback
        self.deselection_callback = deselection_callback