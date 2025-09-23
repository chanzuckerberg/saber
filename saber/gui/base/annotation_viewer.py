from saber.gui.base.segmentation_picker import SegmentationViewer
from typing import Dict, List, Optional
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import cv2


class AnnotationSegmentationViewer(SegmentationViewer):
    """
    A segmentation viewer that tracks annotations by mask values instead of indices.
    """
    
    def __init__(self, image, masks, class_dict, selected_class, annotations_dict, current_run_id):
        """
        :param image: 2D numpy array (Nx, Ny) - the background image
        :param masks: 3D array where masks[i] is the i-th mask, OR
                      2D array where each unique value represents a different mask
        :param class_dict: Dictionary storing class metadata
        :param selected_class: Currently selected class
        :param annotations_dict: Reference to the main annotations dictionary
        :param current_run_id: Current run ID being viewed
        """
        # First, determine if masks is a 2D label map or 3D stack of binary masks
        if len(masks.shape) == 2:
            # It's a 2D label map - extract individual masks
            self.mask_values = np.unique(masks[masks > 0])  # Get all non-zero values
            self.extracted_masks = []
            for val in self.mask_values:
                self.extracted_masks.append((masks == val).astype(np.float32))
            masks = self.extracted_masks
        elif len(masks.shape) == 3:
            # It's already a stack of masks - extract their values
            # Assuming each mask might have a unique value or we assign values based on index
            self.mask_values = []
            for i, mask in enumerate(masks):
                # Try to find the unique non-zero value in this mask
                unique_vals = np.unique(mask[mask > 0])
                if len(unique_vals) > 0:
                    # Use the first non-zero value as the mask's ID
                    self.mask_values.append(unique_vals[0])
                else:
                    # If mask is binary, assign value based on index
                    self.mask_values.append(i + 1)
        
        super().__init__(image, masks)
        
        self.class_dict = class_dict
        self.selected_class = selected_class
        self.annotations_dict = annotations_dict  # Direct reference to main annotations
        self.current_run_id = current_run_id
        
        # Create a mapping from mask index to mask value
        self.index_to_value = {i: val for i, val in enumerate(self.mask_values)}
        self.value_to_index = {val: i for i, val in enumerate(self.mask_values)}
        
        # Store boundary items for each mask
        self.left_boundary_items = []
        self.right_boundary_items = []
        
        # Track which mask is currently highlighted (by value, not index)
        self.highlighted_mask_value = None
        
        # Initialize the viewer
        self.initialize_overlays()
        
        # Load any existing annotations for this run
        self.load_existing_annotations()
    
    def _get_boundary_opencv_fast(self, mask: np.ndarray) -> Optional[np.ndarray]:
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
    
    def initialize_overlays(self):
        """Create overlays and boundaries for all masks."""
        for i, mask in enumerate(self.masks):
            # Create mask overlays
            left_item = pg.ImageItem(self.create_overlay_rgba(mask, i))
            left_item.setOpacity(0.4)
            left_item.setZValue(i + 1)
            self.left_view.addItem(left_item)
            self.left_mask_items.append(left_item)

            right_item = pg.ImageItem(self.create_overlay_rgba(mask, i))
            right_item.setOpacity(0.4)
            right_item.setZValue(i + 1)
            right_item.setVisible(False)
            self.right_view.addItem(right_item)
            self.right_mask_items.append(right_item)
            
            # Create boundary items
            boundary_pts = self._get_boundary_opencv_fast(mask)
            
            # Left boundary (initially hidden)
            left_boundary = pg.PlotDataItem(pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.SolidLine))
            left_boundary.setZValue(1000 + i)
            left_boundary.setVisible(False)
            self.left_view.addItem(left_boundary)
            self.left_boundary_items.append(left_boundary)
            
            # Right boundary (initially hidden)
            right_boundary = pg.PlotDataItem(pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.SolidLine))
            right_boundary.setZValue(1000 + i)
            right_boundary.setVisible(False)
            self.right_view.addItem(right_boundary)
            self.right_boundary_items.append(right_boundary)
            
            # Set boundary data if available
            if boundary_pts is not None and len(boundary_pts) > 0:
                boundary_pts = np.vstack([boundary_pts, boundary_pts[0:1]])
                left_boundary.setData(boundary_pts[:, 1], boundary_pts[:, 0])
                right_boundary.setData(boundary_pts[:, 1], boundary_pts[:, 0])
    
    def load_existing_annotations(self):
        """Load and display any existing annotations for the current run"""
        if self.current_run_id in self.annotations_dict:
            run_annotations = self.annotations_dict[self.current_run_id]
            
            # Clear all class masks first
            for class_name in self.class_dict:
                self.class_dict[class_name]['masks'].clear()
            
            # Restore annotations - now using mask values
            for mask_value_str, class_name in run_annotations.items():
                mask_value = float(mask_value_str)  # Convert back from string
                
                if class_name in self.class_dict and mask_value in self.value_to_index:
                    mask_idx = self.value_to_index[mask_value]
                    
                    # Add mask value (not index) to class dict
                    self.class_dict[class_name]['masks'].append(mask_value)
                    
                    # Update visibility using index
                    self.left_mask_items[mask_idx].setVisible(False)
                    self.right_mask_items[mask_idx].setVisible(True)
                    
                    # Update color
                    updated_overlay = self.create_overlay_rgba(self.masks[mask_idx], class_name=class_name)
                    self.right_mask_items[mask_idx].setImage(updated_overlay)
    
    def highlight_mask(self, mask_value):
        """Highlight a specific mask with boundary on the appropriate panel"""
        # Clear any existing highlight
        self.clear_highlight()
        
        if mask_value not in self.value_to_index:
            return
            
        mask_idx = self.value_to_index[mask_value]
        self.highlighted_mask_value = mask_value
        
        # Determine which panel to show the boundary on
        if mask_idx < len(self.right_mask_items) and self.right_mask_items[mask_idx].isVisible():
            self.right_boundary_items[mask_idx].setVisible(True)
        elif mask_idx < len(self.left_mask_items) and self.left_mask_items[mask_idx].isVisible():
            self.left_boundary_items[mask_idx].setVisible(True)
    
    def clear_highlight(self):
        """Clear all boundary highlights"""
        for boundary in self.left_boundary_items:
            boundary.setVisible(False)
        for boundary in self.right_boundary_items:
            boundary.setVisible(False)
        self.highlighted_mask_value = None
    
    def mouse_clicked(self, event):
        """Handle mouse clicks to accept masks or toggle selection"""
        if not self.selected_class or self.selected_class not in self.class_dict:
            print("No class selected - please add and select a class first")
            return
            
        scene_pos = event.scenePos()
        left_image_pos = self.left_base_img_item.mapFromScene(scene_pos)
        right_image_pos = self.right_base_img_item.mapFromScene(scene_pos)
        
        Nx, Ny = self.image.shape[:2]
        
        # Check if click is in left view
        if self.left_view.sceneBoundingRect().contains(scene_pos):
            x = int(left_image_pos.x())
            y = int(left_image_pos.y())
            
            if not (0 <= x < Nx and 0 <= y < Ny):
                return
            
            # Find which masks contain this pixel and are visible
            mask_hits = []
            for i in range(len(self.masks)):
                if self.masks[i][x, y] > 0 and self.left_mask_items[i].isVisible():
                    mask_hits.append(i)
            
            if not mask_hits:
                return
            
            # Cycle through overlapping masks
            if not hasattr(self, '_last_click_pos') or self._last_click_pos != (x, y):
                self._last_click_pos = (x, y)
                self._current_mask_index = 0
            else:
                self._current_mask_index = (self._current_mask_index + 1) % len(mask_hits)
            
            i_hit = mask_hits[self._current_mask_index]
            mask_value = self.index_to_value[i_hit]
            
            if event.button() == QtCore.Qt.LeftButton:
                # Move mask to right panel
                self.left_mask_items[i_hit].setVisible(False)
                self.right_mask_items[i_hit].setVisible(True)
                self.left_boundary_items[i_hit].setVisible(False)
                
                # Add mask VALUE (not index) to class dict
                self.class_dict[self.selected_class]['masks'].append(mask_value)
                
                # Update color with class color
                updated_overlay = self.create_overlay_rgba(self.masks[i_hit], class_name=self.selected_class)
                self.right_mask_items[i_hit].setImage(updated_overlay)
                
                # Update annotations dictionary with mask VALUE
                if self.current_run_id not in self.annotations_dict:
                    self.annotations_dict[self.current_run_id] = {}
                self.annotations_dict[self.current_run_id][str(mask_value)] = self.selected_class
                
                # Highlight the newly accepted mask
                self.highlight_mask(mask_value)
                
                # print(f"Added: Run {self.current_run_id}, Mask Value {mask_value} -> {self.selected_class}")
        
        # Check if click is in right view
        elif self.right_view.sceneBoundingRect().contains(scene_pos):
            x = int(right_image_pos.x())
            y = int(right_image_pos.y())
            
            if not (0 <= x < Nx and 0 <= y < Ny):
                return
            
            # Find which masks contain this pixel and are visible
            for i in range(len(self.masks)):
                if self.masks[i][x, y] > 0 and self.right_mask_items[i].isVisible():
                    mask_value = self.index_to_value[i]
                    
                    if event.button() == QtCore.Qt.LeftButton:
                        # Toggle selection/highlight
                        if self.highlighted_mask_value == mask_value:
                            self.clear_highlight()
                            # print(f"Deselected mask value {mask_value}")
                        else:
                            self.highlight_mask(mask_value)
                            # print(f"Selected mask value {mask_value}")
                        break
    
    def keyPressEvent(self, event):
        """Handle 'R' key to remove the currently highlighted mask"""
        if event.key() == QtCore.Qt.Key_R:
            if self.highlighted_mask_value is None:
                print("No mask selected to remove")
                return
            
            mask_value = self.highlighted_mask_value
            mask_idx = self.value_to_index[mask_value]
            
            # Check if this mask is actually on the right panel
            if mask_idx < len(self.right_mask_items) and self.right_mask_items[mask_idx].isVisible():
                # Find which class this mask belongs to
                class_name = None
                if self.current_run_id in self.annotations_dict:
                    mask_key = str(mask_value)
                    if mask_key in self.annotations_dict[self.current_run_id]:
                        class_name = self.annotations_dict[self.current_run_id][mask_key]
                        del self.annotations_dict[self.current_run_id][mask_key]
                
                # Remove mask VALUE from class dict
                if class_name and class_name in self.class_dict:
                    if mask_value in self.class_dict[class_name]['masks']:
                        self.class_dict[class_name]['masks'].remove(mask_value)
                
                # Clear the highlight first
                self.clear_highlight()
                
                # Move mask back to left panel
                self.left_mask_items[mask_idx].setVisible(True)
                self.right_mask_items[mask_idx].setVisible(False)
                
                # print(f"Removed mask value {mask_value} (class: {class_name})")
            # else:
            #     print(f"Selected mask value {mask_value} is not on the right panel")
        else:
            super().keyPressEvent(event)
    
    def create_overlay_rgba(self, mask, index=0, class_name=None):
        """Create colored overlay for mask"""
        Nx, Ny = mask.shape
        rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)
        
        if class_name is not None and class_name in self.class_dict:
            index = self.class_dict[class_name]['value'] - 1
        
        color = self.tab10_colors[index % len(self.tab10_colors)]
        
        inds = mask > 0.5
        rgba[inds, 0] = color[0]
        rgba[inds, 1] = color[1]
        rgba[inds, 2] = color[2]
        rgba[inds, 3] = 1.0
        
        return rgba
    
    def update_current_run(self, run_id):
        """Update the current run ID"""
        self.current_run_id = run_id
    
    def load_data(self, base_image, masks, class_dict, run_id):
        """Load new data and update run ID"""
        self.current_run_id = run_id
        self.base_image = base_image
        self.masks = masks
        self.class_dict = class_dict
        
        # Determine mask values for the new data
        if len(masks.shape) == 2:
            self.mask_values = np.unique(masks[masks > 0])
            self.extracted_masks = []
            for val in self.mask_values:
                self.extracted_masks.append((masks == val).astype(np.float32))
            self.masks = self.extracted_masks
        elif len(masks.shape) == 3:
            self.mask_values = []
            for i, mask in enumerate(self.masks):
                unique_vals = np.unique(mask[mask > 0])
                if len(unique_vals) > 0:
                    self.mask_values.append(unique_vals[0])
                else:
                    self.mask_values.append(i + 1)
        
        # Update mappings
        self.index_to_value = {i: val for i, val in enumerate(self.mask_values)}
        self.value_to_index = {val: i for i, val in enumerate(self.mask_values)}
        
        # Clear indices for all classes
        for class_name in self.class_dict.keys():
            self.class_dict[class_name]['masks'].clear()
        
        # Update base images
        self.left_base_img_item.setImage(self.base_image)
        self.right_base_img_item.setImage(self.base_image)
        
        # Clear old masks and boundaries
        for item in self.left_mask_items:
            self.left_view.removeItem(item)
        for item in self.right_mask_items:
            self.right_view.removeItem(item)
        for item in self.left_boundary_items:
            self.left_view.removeItem(item)
        for item in self.right_boundary_items:
            self.right_view.removeItem(item)
        
        self.left_mask_items.clear()
        self.right_mask_items.clear()
        self.left_boundary_items.clear()
        self.right_boundary_items.clear()
        
        # Reinitialize overlays
        self.initialize_overlays()
        
        # Load existing annotations
        self.load_existing_annotations()