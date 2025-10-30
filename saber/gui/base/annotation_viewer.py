from saber.gui.base.segmentation_picker import SegmentationViewer
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import cv2
from typing import Dict, List, Optional


class AnnotationSegmentationViewer(SegmentationViewer):
    """
    Optimized segmentation viewer that tracks annotations by mask values.
    """
    
    def __init__(self, image, masks, class_dict, selected_class, annotations_dict, current_run_id):
        # Extract mask values before calling super().__init__()
        mask_values, extracted_masks = self._extract_mask_values_static(masks)
        self.mask_values = mask_values
        
        # Pass the extracted masks to parent
        super().__init__(image, extracted_masks if extracted_masks is not None else masks)

        # Link the views for synchronized zoom/pan
        self.right_view.setXLink(self.left_view)
        self.right_view.setYLink(self.left_view)
        
        self.class_dict = class_dict
        self.selected_class = selected_class
        self.annotations_dict = annotations_dict
        self.current_run_id = current_run_id
        
        self.index_to_value = {i: val for i, val in enumerate(self.mask_values)}
        self.value_to_index = {val: i for i, val in enumerate(self.mask_values)}
        
        self.left_boundary_items = []
        self.right_boundary_items = []
        self.highlighted_mask_value = None
        
        self.initialize_overlays()
        self.load_existing_annotations()
    
    @staticmethod
    def _extract_mask_values_static(masks):
        """Vectorized mask extraction from 2D label maps - static version for __init__"""
        if len(masks.shape) == 2:
            # Vectorized approach - much faster than looping
            mask_values = np.unique(masks[masks > 0])
            num_masks = len(mask_values)
            
            # Create all masks at once using broadcasting
            masks_3d = masks[np.newaxis, :, :] == mask_values[:, np.newaxis, np.newaxis]
            extracted_masks = [masks_3d[i].astype(np.float32) for i in range(num_masks)]
            return mask_values, extracted_masks
        elif len(masks.shape) == 3:
            mask_values = []
            for i, mask in enumerate(masks):
                unique_vals = np.unique(mask[mask > 0])
                if len(unique_vals) > 0:
                    mask_values.append(unique_vals[0])
                else:
                    mask_values.append(i + 1)
            return np.array(mask_values), list(masks)
        return np.array([]), []
    
    def _extract_mask_values(self, masks):
        """Instance method wrapper for load_data"""
        self.mask_values, extracted_masks = self._extract_mask_values_static(masks)
        if extracted_masks:
            self.masks = extracted_masks
    
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
        """Create overlays and boundaries for all masks - deferred rendering"""
        num_masks = len(self.masks)
        
        # Create items only if needed, otherwise reuse
        current_count = len(self.left_mask_items)
        
        if current_count < num_masks:
            # Need to create more items
            for i in range(current_count, num_masks):
                left_item = pg.ImageItem()
                left_item.setOpacity(0.4)
                left_item.setZValue(i + 1)
                self.left_view.addItem(left_item)
                self.left_mask_items.append(left_item)

                right_item = pg.ImageItem()
                right_item.setOpacity(0.4)
                right_item.setZValue(i + 1)
                right_item.setVisible(False)
                self.right_view.addItem(right_item)
                self.right_mask_items.append(right_item)
                
                left_boundary = pg.PlotDataItem(pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.SolidLine))
                left_boundary.setZValue(1000 + i)
                left_boundary.setVisible(False)
                self.left_view.addItem(left_boundary)
                self.left_boundary_items.append(left_boundary)
                
                right_boundary = pg.PlotDataItem(pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.SolidLine))
                right_boundary.setZValue(1000 + i)
                right_boundary.setVisible(False)
                self.right_view.addItem(right_boundary)
                self.right_boundary_items.append(right_boundary)
        elif current_count > num_masks:
            # Hide/remove excess items
            for i in range(num_masks, current_count):
                self.left_mask_items[i].setVisible(False)
                self.right_mask_items[i].setVisible(False)
                self.left_boundary_items[i].setVisible(False)
                self.right_boundary_items[i].setVisible(False)
        
        # Reset visibility for used items
        for i in range(min(num_masks, current_count)):
            self.left_mask_items[i].setVisible(True)
            self.right_mask_items[i].setVisible(False)
            self.left_boundary_items[i].setVisible(False)
            self.right_boundary_items[i].setVisible(False)
        
        # Render visible masks on left panel (initial display)
        self._render_left_visible_masks()

    def _render_left_visible_masks(self):
        """Render RGBA overlays only for visible masks on the left panel."""
        if not hasattr(self, '_rgba_cache'):
            self._rgba_cache = {}
        
        for i in range(len(self.masks)):
            if i < len(self.left_mask_items) and self.left_mask_items[i].isVisible():
                if i not in self._rgba_cache:
                    rgba = self.create_overlay_rgba(self.masks[i], i)
                    self._rgba_cache[i] = rgba
                self.left_mask_items[i].setImage(self._rgba_cache[i])

    def _render_visible_masks(self):
        """Render RGBA overlays only for currently visible masks."""
        if not hasattr(self, '_rgba_cache'):
            self._rgba_cache = {}
        
        for i in range(len(self.masks)):
            if self.left_mask_items[i].isVisible() and i not in self._rgba_cache:
                rgba = self.create_overlay_rgba(self.masks[i], i)
                self._rgba_cache[i] = rgba
                self.left_mask_items[i].setImage(rgba)
            elif self.left_mask_items[i].isVisible() and i in self._rgba_cache:
                self.left_mask_items[i].setImage(self._rgba_cache[i])

    def _ensure_mask_rendered(self, mask_idx, class_name=None):
        """Ensure a specific mask has its RGBA overlay created."""
        if not hasattr(self, '_rgba_cache'):
            self._rgba_cache = {}
        
        cache_key = (mask_idx, class_name) if class_name else mask_idx
        
        if cache_key not in self._rgba_cache:
            rgba = self.create_overlay_rgba(self.masks[mask_idx], mask_idx, class_name=class_name)
            self._rgba_cache[cache_key] = rgba
        
        return self._rgba_cache[cache_key]
    
    def load_existing_annotations(self):
        """Load and display existing annotations - batch visibility updates"""
        if self.current_run_id not in self.annotations_dict:
            return
            
        run_annotations = self.annotations_dict[self.current_run_id]
        
        # Clear all class masks first
        for class_name in self.class_dict:
            self.class_dict[class_name]['masks'].clear()
        
        # Batch visibility changes
        to_hide_left = []
        to_show_right = []
        to_render = []
        
        for mask_value_str, class_name in run_annotations.items():
            mask_value = float(mask_value_str)
            
            if class_name in self.class_dict and mask_value in self.value_to_index:
                mask_idx = self.value_to_index[mask_value]
                
                self.class_dict[class_name]['masks'].append(mask_value)
                
                to_hide_left.append(mask_idx)
                to_show_right.append((mask_idx, class_name))
        
        # Apply visibility changes in batch
        for mask_idx in to_hide_left:
            self.left_mask_items[mask_idx].setVisible(False)
        
        for mask_idx, class_name in to_show_right:
            self.right_mask_items[mask_idx].setVisible(True)
            # Render on demand
            updated_overlay = self._ensure_mask_rendered(mask_idx, class_name=class_name)
            self.right_mask_items[mask_idx].setImage(updated_overlay)
    
    def _compute_boundary_if_needed(self, mask_idx):
        """Compute boundary for a mask if not already done"""
        if not hasattr(self, '_boundary_cache'):
            self._boundary_cache = {}
        
        if mask_idx not in self._boundary_cache:
            boundary_pts = self._get_boundary_opencv_fast(self.masks[mask_idx])
            if boundary_pts is not None and len(boundary_pts) > 0:
                boundary_pts = np.vstack([boundary_pts, boundary_pts[0:1]])
                self._boundary_cache[mask_idx] = boundary_pts
            else:
                self._boundary_cache[mask_idx] = None
        
        return self._boundary_cache[mask_idx]

    def highlight_mask(self, mask_value):
        """Highlight a specific mask with boundary on the appropriate panel"""
        self.clear_highlight()
        
        if mask_value not in self.value_to_index:
            return
            
        mask_idx = self.value_to_index[mask_value]
        self.highlighted_mask_value = mask_value
        
        boundary_pts = self._compute_boundary_if_needed(mask_idx)
        if boundary_pts is not None:
            if mask_idx < len(self.right_mask_items) and self.right_mask_items[mask_idx].isVisible():
                self.right_boundary_items[mask_idx].setData(boundary_pts[:, 1], boundary_pts[:, 0])
                self.right_boundary_items[mask_idx].setVisible(True)
            elif mask_idx < len(self.left_mask_items) and self.left_mask_items[mask_idx].isVisible():
                self.left_boundary_items[mask_idx].setData(boundary_pts[:, 1], boundary_pts[:, 0])
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
        
        if self.left_view.sceneBoundingRect().contains(scene_pos):
            x = int(left_image_pos.x())
            y = int(left_image_pos.y())
            
            if not (0 <= x < Nx and 0 <= y < Ny):
                return
            
            mask_hits = []
            for i in range(len(self.masks)):
                if self.masks[i][x, y] > 0 and self.left_mask_items[i].isVisible():
                    mask_hits.append(i)
            
            if not mask_hits:
                return
            
            if not hasattr(self, '_last_click_pos') or self._last_click_pos != (x, y):
                self._last_click_pos = (x, y)
                self._current_mask_index = 0
            else:
                self._current_mask_index = (self._current_mask_index + 1) % len(mask_hits)
            
            i_hit = mask_hits[self._current_mask_index]
            mask_value = self.index_to_value[i_hit]
            
            if event.button() == QtCore.Qt.LeftButton:
                # Render mask before showing it
                updated_overlay = self._ensure_mask_rendered(i_hit, class_name=self.selected_class)
                
                self.left_mask_items[i_hit].setVisible(False)
                self.right_mask_items[i_hit].setImage(updated_overlay)
                self.right_mask_items[i_hit].setVisible(True)
                self.left_boundary_items[i_hit].setVisible(False)
                
                self.class_dict[self.selected_class]['masks'].append(mask_value)
                
                if self.current_run_id not in self.annotations_dict:
                    self.annotations_dict[self.current_run_id] = {}
                self.annotations_dict[self.current_run_id][str(mask_value)] = self.selected_class
                
                self.highlight_mask(mask_value)
        
        elif self.right_view.sceneBoundingRect().contains(scene_pos):
            x = int(right_image_pos.x())
            y = int(right_image_pos.y())
            
            if not (0 <= x < Nx and 0 <= y < Ny):
                return
            
            for i in range(len(self.masks)):
                if self.masks[i][x, y] > 0 and self.right_mask_items[i].isVisible():
                    mask_value = self.index_to_value[i]
                    
                    if event.button() == QtCore.Qt.LeftButton:
                        if self.highlighted_mask_value == mask_value:
                            self.clear_highlight()
                        else:
                            self.highlight_mask(mask_value)
                        break
    
    def keyPressEvent(self, event):
        """Handle 'R' key to remove the currently highlighted mask"""
        if event.key() == QtCore.Qt.Key_R:
            if self.highlighted_mask_value is None:
                print("No mask selected to remove")
                return
            
            mask_value = self.highlighted_mask_value
            mask_idx = self.value_to_index[mask_value]
            
            if mask_idx < len(self.right_mask_items) and self.right_mask_items[mask_idx].isVisible():
                class_name = None
                if self.current_run_id in self.annotations_dict:
                    mask_key = str(mask_value)
                    if mask_key in self.annotations_dict[self.current_run_id]:
                        class_name = self.annotations_dict[self.current_run_id][mask_key]
                        del self.annotations_dict[self.current_run_id][mask_key]
                
                if class_name and class_name in self.class_dict:
                    if mask_value in self.class_dict[class_name]['masks']:
                        self.class_dict[class_name]['masks'].remove(mask_value)
                
                self.clear_highlight()
                
                # Render original color before showing
                if mask_idx not in self._rgba_cache:
                    rgba = self.create_overlay_rgba(self.masks[mask_idx], mask_idx)
                    self._rgba_cache[mask_idx] = rgba
                    self.left_mask_items[mask_idx].setImage(rgba)
                
                self.left_mask_items[mask_idx].setVisible(True)
                self.right_mask_items[mask_idx].setVisible(False)
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
        """Load new data with minimal overhead - reuse existing items"""
        self.current_run_id = run_id
        self.base_image = base_image
        self.class_dict = class_dict
        
        # Extract mask values efficiently
        self._extract_mask_values(masks)
        
        # Update mappings
        self.index_to_value = {i: val for i, val in enumerate(self.mask_values)}
        self.value_to_index = {val: i for i, val in enumerate(self.mask_values)}
        
        # Clear class masks
        for class_name in self.class_dict.keys():
            self.class_dict[class_name]['masks'].clear()
        
        # Update base images
        self.left_base_img_item.setImage(self.base_image)
        self.right_base_img_item.setImage(self.base_image)
        
        # Clear caches for new data
        if hasattr(self, '_boundary_cache'):
            self._boundary_cache.clear()
        if hasattr(self, '_rgba_cache'):
            self._rgba_cache.clear()
        
        # Reuse items instead of destroying/recreating
        self.initialize_overlays()
        
        # Load existing annotations
        self.load_existing_annotations()