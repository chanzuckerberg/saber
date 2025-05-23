from saber.gui.segmentation_picker import SegmentationViewer
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import sys

class MultiClassSegmentationViewer(SegmentationViewer):
    def __init__(self, image, masks, class_dict, selected_class):
        """
        A segmentation viewer that supports multi-class masks.

        :param image: 2D numpy array (Nx, Ny) - the background image
        :param masks: List (or array) of 2D masks (shape: (N_classes, Nx, Ny))
        :param class_dict: Dictionary storing class metadata (value, masks, color)
        :param selected_class: The currently selected class
        """
        super().__init__(image, masks)

        # Handle additional arguments specific to MultiClassSegmentationViewer
        self.class_dict = class_dict
        self.selected_class = selected_class
 
        # Call initialize_overlays after class_dict is set
        self.initialize_overlays()

    def mouse_clicked(self, event):
        """
        Handle mouse clicks to accept/reject masks and assign them to the current class.
        """
        # Call the base class method to get the clicked mask index
        scene_pos = event.scenePos()
        image_pos = self.left_base_img_item.mapFromScene(scene_pos)
        
        # Get the image dimensions
        Nx, Ny = self.image.shape[:2]
        
        # Convert to array coordinates, accounting for the image orientation
        x = int(image_pos.x())
        y = int(image_pos.y())

        # Check bounds
        if not (0 <= x < Nx and 0 <= y < Ny):
            print(f"Clicked out of bounds: {x}, {y}")
            return  # clicked out of bounds        

        mask_hits = []
        for i in range(len(self.masks)):
            if self.masks[i][x, y] > 0:
                mask_hits.append(i)

        if not mask_hits:
            return  # clicked on empty space in the left image

        # If this is the first click at this position, store the mask hits
        if not hasattr(self, '_last_click_pos') or self._last_click_pos != (x, y):
            self._last_click_pos = (x, y)
            self._current_mask_index = 0
        else:
            # Cycle to the next mask
            self._current_mask_index = (self._current_mask_index + 1) % len(mask_hits)

        # Get the current mask to select
        i_hit = mask_hits[self._current_mask_index]

        if event.button() == QtCore.Qt.LeftButton:
            # Hide the mask on the left and show it on the right
            if self.left_mask_items[i_hit].isVisible():
                self.left_mask_items[i_hit].setVisible(False)
                self.right_mask_items[i_hit].setVisible(True)

                # Add to the class_dict for the current class
                class_name = self.selected_class
                self.class_dict[self.selected_class]['masks'].append(i_hit)
                
                # Update the mask's overlay with the new class color
                updated_overlay = self.create_overlay_rgba(self.masks[i_hit], class_name=class_name)
                self.right_mask_items[i_hit].setImage(updated_overlay)

                print(f"Accepted mask {i_hit} for class {self.selected_class}")
                print(f"Class {self.selected_class} now has {len(self.class_dict[self.selected_class]['masks'])} masks.")

        print(f"Current accepted masks: {self.class_dict[self.selected_class]['masks']}") 

    def create_overlay_rgba(self, mask, index=0, class_name=None):
        """
        Convert a single 2D mask (Nx x Ny) into an RGBA overlay.
        Use the integer value from class_dict as the color index.

        :param mask: 2D numpy array representing the mask
        :param class_name: The class name for the mask
        """
        Nx, Ny = mask.shape
        rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)

        # Get the index for the class name
        # Use default index and color if no class is assigned
        if class_name is not None and class_name in self.class_dict:
            index = self.class_dict[class_name]['value'] - 1  # Adjust for 0-based indexing
        
        # Get the color corresponding to the class
        color = self.tab10_colors[index % len(self.tab10_colors)]

        # Apply the color to the mask
        inds = mask > 0.5
        rgba[inds, 0] = color[0]  # Red channel
        rgba[inds, 1] = color[1]  # Green channel
        rgba[inds, 2] = color[2]  # Blue channel
        rgba[inds, 3] = 1.0       # Alpha channel for transparency

        return rgba
    
    def keyPressEvent(self, event):
        """
        Press 'r' => undo the last accepted mask for the selected class:
            - Remove it from the class_dict for the selected class
            - Show it on the left view
            - Hide it on the right view
        """
        key = event.key()
        if key == QtCore.Qt.Key_R:
            if self.selected_class not in self.class_dict:
                print(f"Selected class '{self.selected_class}' is invalid.")
                return

            # Check if there are any masks to undo for the selected class
            class_masks = self.class_dict[self.selected_class]['masks']
            if not class_masks:
                print(f"No masks to undo for class '{self.selected_class}'.")
                return

            # Undo the most recently assigned mask for the selected class
            last_index = class_masks.pop()  # Remove the last mask from the list

            # Update visibility: show it on the left, hide it on the right
            self.left_mask_items[last_index].setVisible(True)
            self.right_mask_items[last_index].setVisible(False)

            print(f"Undid mask {last_index} for class '{self.selected_class}': now visible on left.")
        else:
            super().keyPressEvent(event)    
