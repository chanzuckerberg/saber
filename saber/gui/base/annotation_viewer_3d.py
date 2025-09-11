from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from typing import Optional, Dict, Any
from saber.gui.base.utils import (
    get_boundary_opencv_fast,
    create_overlay_rgba,
    extract_masks_from_labels,
    process_mask_data,
    create_mask_overlay_item,
    create_boundary_item,
    highlight_mask_with_boundary,
    clear_all_highlights
)


class AnnotationSegmentationViewer3D(QtWidgets.QWidget):
    """
    A standalone 3D segmentation viewer with slice navigation.
    This does NOT inherit from AnnotationSegmentationViewer to avoid widget hierarchy issues.
    """
    
    def __init__(self, volume, masks, class_dict, selected_class, annotations_dict, current_run_id):
        """
        :param volume: 3D numpy array (Nz, Nx, Ny) - the background volume
        :param masks: 4D array where masks[i] is the i-th 3D mask (Nz, Nx, Ny), OR
                      3D array where each unique value represents a different mask
        :param class_dict: Dictionary storing class metadata
        :param selected_class: Currently selected class
        :param annotations_dict: Reference to the main annotations dictionary
        :param current_run_id: Current run ID being viewed
        """
        super().__init__()
        
        # Store parameters
        self.class_dict = class_dict
        self.selected_class = selected_class
        self.annotations_dict = annotations_dict
        self.current_run_id = current_run_id
        
        # Store 3D data
        self.volume_3d = volume
        self.masks_3d = masks
        self.current_slice = int(volume.shape[0] / 2)
        self.n_slices = volume.shape[0]
        
        # Extract mask values from 3D data
        if len(masks.shape) == 3:
            # 3D label map
            self.all_mask_values = np.unique(masks[masks > 0])
            self.is_label_map = True
        elif len(masks.shape) == 4:
            # Stack of 3D masks
            self.all_mask_values = []
            for i, mask_vol in enumerate(masks):
                unique_vals = np.unique(mask_vol[mask_vol > 0])
                if len(unique_vals) > 0:
                    self.all_mask_values.append(unique_vals[0])
                else:
                    self.all_mask_values.append(i + 1)
            self.is_label_map = False
        else:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")
        
        # Get initial slice data
        self.image = volume[self.current_slice]
        initial_masks = self._get_slice_masks(self.current_slice)
        
        # Process the initial masks
        if len(initial_masks.shape) == 2:
            self.masks, self.mask_values = extract_masks_from_labels(initial_masks)
        else:
            self.masks = [m.astype(np.float32) for m in initial_masks]
            self.mask_values = self.all_mask_values
        
        # Create mappings
        self.index_to_value = {i: val for i, val in enumerate(self.all_mask_values)}
        self.value_to_index = {val: i for i, val in enumerate(self.all_mask_values)}
        
        # Track which mask is currently highlighted
        self.highlighted_mask_value = None
        
        # For click cycling through overlapping masks
        self._last_click_pos = None
        self._current_mask_index = 0
        
        # Storage for mask and boundary items
        self.left_mask_items = []
        self.right_mask_items = []
        self.left_boundary_items = []
        self.right_boundary_items = []
        
        # Setup the UI
        self._setup_ui()
        
        # Initialize overlays
        self.initialize_overlays()
        
        # Load any existing annotations
        self.load_existing_annotations()
    
    def _setup_ui(self):
        """Setup the complete UI with dual panels and slice control"""
        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        # Create the graphics widget (like the parent class would have)
        self.graphics_widget = pg.GraphicsLayoutWidget()
        
        # Add two views side by side with centered alignment
        self.left_view = self.graphics_widget.addViewBox(row=0, col=0)
        self.right_view = self.graphics_widget.addViewBox(row=0, col=1)
        
        # Link the views for synchronized navigation
        self.left_view.setXLink(self.right_view)
        self.left_view.setYLink(self.right_view)
        
        # Lock aspect ratios
        self.left_view.setAspectLocked(True)
        self.right_view.setAspectLocked(True)
        
        # Add base images
        self.left_base_img_item = pg.ImageItem(self.image)
        self.right_base_img_item = pg.ImageItem(self.image)
        self.left_view.addItem(self.left_base_img_item)
        self.right_view.addItem(self.right_base_img_item)
        
        # Auto-range to fit the images and center them
        self.left_view.autoRange()
        self.right_view.autoRange()
        
        # Connect mouse click event
        self.graphics_widget.scene().sigMouseClicked.connect(self.mouse_clicked)
        
        # Add graphics widget to layout
        main_layout.addWidget(self.graphics_widget, stretch=1)
        
        # Create and add slice control
        self._create_slice_control()
        main_layout.addWidget(self.slider_widget)
    
    def _create_slice_control(self):
        """Create slider widget for slice navigation"""
        self.slider_widget = QtWidgets.QWidget()
        slider_layout = QtWidgets.QVBoxLayout()
        slider_layout.setContentsMargins(20, 10, 5, 10)  # Add top/bottom margins
        slider_layout.setSpacing(5)
        
        # Top label showing maximum slice
        top_label = QtWidgets.QLabel(str(self.n_slices))
        top_label.setAlignment(QtCore.Qt.AlignCenter)
        top_label.setStyleSheet("color: gray; font-size: 10px;")
        
        # Slice slider - will expand to fill available space
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.n_slices - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.setTickPosition(QtWidgets.QSlider.TicksRight)
        self.slice_slider.setTickInterval(max(1, self.n_slices // 10))
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        # Make slider inverted so top = last slice, bottom = first slice
        self.slice_slider.setInvertedAppearance(True)
        
        # Bottom label showing slice 1
        bottom_label = QtWidgets.QLabel("1")
        bottom_label.setAlignment(QtCore.Qt.AlignCenter)
        bottom_label.setStyleSheet("color: gray; font-size: 10px;")
        
        # Current slice label (larger, bold)
        self.slice_label = QtWidgets.QLabel(f'{self.current_slice + 1}')
        self.slice_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slice_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        # Add widgets to layout
        slider_layout.addWidget(self.slice_label)
        slider_layout.addWidget(top_label)
        slider_layout.addWidget(self.slice_slider, stretch=1)  # Slider expands
        slider_layout.addWidget(bottom_label)
        
        self.slider_widget.setLayout(slider_layout)
        
        # Set fixed width for slider widget
        self.slider_widget.setMaximumWidth(80)
        self.slider_widget.setMinimumWidth(10)
    
    def _get_slice_masks(self, slice_idx):
        """Extract 2D masks for a specific slice"""
        if self.is_label_map:
            return self.masks_3d[slice_idx]
        else:
            masks_2d = []
            for mask_vol in self.masks_3d:
                masks_2d.append(mask_vol[slice_idx])
            return np.array(masks_2d)
    
    def initialize_overlays(self):
        """Create overlays and boundaries for all masks"""
        for i, mask in enumerate(self.masks):
            has_content = mask.any()
            
            # Create mask overlays
            left_item = create_mask_overlay_item(mask, i, self.class_dict)
            left_item.setVisible(has_content)
            self.left_view.addItem(left_item)
            self.left_mask_items.append(left_item)
            
            right_item = create_mask_overlay_item(mask, i, self.class_dict)
            right_item.setVisible(False)
            self.right_view.addItem(right_item)
            self.right_mask_items.append(right_item)
            
            # Create boundary items
            left_boundary = create_boundary_item(mask, i)
            self.left_view.addItem(left_boundary)
            self.left_boundary_items.append(left_boundary)
            
            right_boundary = create_boundary_item(mask, i)
            self.right_view.addItem(right_boundary)
            self.right_boundary_items.append(right_boundary)
    
    def load_existing_annotations(self):
        """Load and display any existing annotations for the current run"""
        if self.current_run_id not in self.annotations_dict:
            return
        
        run_annotations = self.annotations_dict[self.current_run_id]
        
        # Clear all class masks first
        for class_name in self.class_dict:
            self.class_dict[class_name]['masks'].clear()
        
        # Restore annotations
        for mask_value_str, class_name in run_annotations.items():
            mask_value = float(mask_value_str)
            
            if class_name in self.class_dict and mask_value in self.value_to_index:
                mask_idx = self.value_to_index[mask_value]
                
                # Add mask value to class dict
                self.class_dict[class_name]['masks'].append(mask_value)
                
                # Check if this mask exists in current slice
                if mask_idx < len(self.masks) and self.masks[mask_idx].any():
                    # Update visibility
                    self.left_mask_items[mask_idx].setVisible(False)
                    self.right_mask_items[mask_idx].setVisible(True)
                    
                    # Update color
                    updated_overlay = create_overlay_rgba(
                        self.masks[mask_idx], mask_idx, self.class_dict, class_name
                    )
                    self.right_mask_items[mask_idx].setImage(updated_overlay)
    
    def on_slice_changed(self, value):
        """Handle slice slider changes"""
        self.current_slice = value
        self.slice_label.setText(f'{self.current_slice + 1}')
        self.update_slice_display()
    
    def update_slice_display(self):
        """Update the display for the current slice"""
        # Clear highlight
        self.clear_highlight()
        
        # Get current slice data
        current_image = self.volume_3d[self.current_slice]
        current_masks = self._get_slice_masks(self.current_slice)
        
        # Update base images
        self.image = current_image
        self.left_base_img_item.setImage(current_image)
        self.right_base_img_item.setImage(current_image)
        
        # Process masks for current slice
        if len(current_masks.shape) == 2:
            self.mask_values = np.unique(current_masks[current_masks > 0])
            self.masks = []
            for val in self.all_mask_values:
                if val in self.mask_values:
                    self.masks.append((current_masks == val).astype(np.float32))
                else:
                    self.masks.append(np.zeros_like(current_masks, dtype=np.float32))
        else:
            self.masks = [m.astype(np.float32) for m in current_masks]
            self.mask_values = self.all_mask_values
        
        # Clear old overlays and boundaries
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
        
        # Restore annotations for this slice
        self._restore_slice_annotations()
        
        # Re-center the views after updating
        self.left_view.autoRange()
        self.right_view.autoRange()
    
    def _restore_slice_annotations(self):
        """Restore annotation visibility for current slice"""
        if self.current_run_id not in self.annotations_dict:
            return
        
        run_annotations = self.annotations_dict[self.current_run_id]
        
        for mask_value_str, class_name in run_annotations.items():
            mask_value = float(mask_value_str)
            
            if mask_value in self.value_to_index and class_name in self.class_dict:
                mask_idx = self.value_to_index[mask_value]
                
                if mask_idx < len(self.masks) and self.masks[mask_idx].any():
                    self.left_mask_items[mask_idx].setVisible(False)
                    self.right_mask_items[mask_idx].setVisible(True)
                    
                    updated_overlay = create_overlay_rgba(
                        self.masks[mask_idx], mask_idx, self.class_dict, class_name
                    )
                    self.right_mask_items[mask_idx].setImage(updated_overlay)
    
    def highlight_mask(self, mask_value):
        """Highlight a specific mask with boundary"""
        self.clear_highlight()
        
        if mask_value not in self.value_to_index:
            return
        
        mask_idx = self.value_to_index[mask_value]
        self.highlighted_mask_value = mask_value
        
        highlight_mask_with_boundary(
            mask_idx, self.left_mask_items, self.right_mask_items,
            self.left_boundary_items, self.right_boundary_items
        )
    
    def clear_highlight(self):
        """Clear all boundary highlights"""
        clear_all_highlights(self.left_boundary_items, self.right_boundary_items)
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
            if self._last_click_pos != (x, y):
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
                
                # Add mask value to class dict
                self.class_dict[self.selected_class]['masks'].append(mask_value)
                
                # Update color with class color
                updated_overlay = create_overlay_rgba(
                    self.masks[i_hit], i_hit, self.class_dict, self.selected_class
                )
                self.right_mask_items[i_hit].setImage(updated_overlay)
                
                # Update annotations
                if self.current_run_id not in self.annotations_dict:
                    self.annotations_dict[self.current_run_id] = {}
                self.annotations_dict[self.current_run_id][str(mask_value)] = self.selected_class
                
                # Highlight the newly accepted mask
                self.highlight_mask(mask_value)
        
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
                        else:
                            self.highlight_mask(mask_value)
                        break
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == QtCore.Qt.Key_R:
            # Remove highlighted mask
            if self.highlighted_mask_value is None:
                print("No mask selected to remove")
                return
            
            mask_value = self.highlighted_mask_value
            mask_idx = self.value_to_index[mask_value]
            
            if mask_idx < len(self.right_mask_items) and self.right_mask_items[mask_idx].isVisible():
                # Find which class this mask belongs to
                class_name = None
                if self.current_run_id in self.annotations_dict:
                    mask_key = str(mask_value)
                    if mask_key in self.annotations_dict[self.current_run_id]:
                        class_name = self.annotations_dict[self.current_run_id][mask_key]
                        del self.annotations_dict[self.current_run_id][mask_key]
                
                # Remove mask value from class dict
                if class_name and class_name in self.class_dict:
                    if mask_value in self.class_dict[class_name]['masks']:
                        self.class_dict[class_name]['masks'].remove(mask_value)
                
                # Clear the highlight first
                self.clear_highlight()
                
                # Move mask back to left panel
                self.left_mask_items[mask_idx].setVisible(True)
                self.right_mask_items[mask_idx].setVisible(False)
                
        elif event.key() == QtCore.Qt.Key_Up:
            # Navigate slices up
            if self.current_slice < self.n_slices - 1:
                self.slice_slider.setValue(self.current_slice + 1)
        elif event.key() == QtCore.Qt.Key_Down:
            # Navigate slices down
            if self.current_slice > 0:
                self.slice_slider.setValue(self.current_slice - 1)
        else:
            super().keyPressEvent(event)
    
    def load_data(self, base_image, masks, class_dict, run_id):
        """Load new data and update run ID"""
        self.current_run_id = run_id
        self.volume_3d = base_image
        self.masks_3d = masks
        self.class_dict = class_dict
        self.n_slices = base_image.shape[0]
        
        # Reset slice to middle
        self.current_slice = int(self.n_slices / 2)
        self.slice_slider.setValue(int(self.n_slices / 2))
        self.slice_slider.setMaximum(self.n_slices - 1)
        
        # Re-extract mask values for the new data
        if len(masks.shape) == 3:
            self.all_mask_values = np.unique(masks[masks > 0])
            self.is_label_map = True
        elif len(masks.shape) == 4:
            self.all_mask_values = []
            for i, mask_vol in enumerate(masks):
                unique_vals = np.unique(mask_vol[mask_vol > 0])
                if len(unique_vals) > 0:
                    self.all_mask_values.append(unique_vals[0])
                else:
                    self.all_mask_values.append(i + 1)
            self.is_label_map = False
        
        # Update mappings
        self.index_to_value = {i: val for i, val in enumerate(self.all_mask_values)}
        self.value_to_index = {val: i for i, val in enumerate(self.all_mask_values)}
        
        # Clear class masks
        for class_name in self.class_dict.keys():
            self.class_dict[class_name]['masks'].clear()
        
        # Update the display
        self.update_slice_display()
        
        # Load existing annotations
        self.load_existing_annotations()