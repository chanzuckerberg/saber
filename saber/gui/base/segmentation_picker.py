from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import sys

class SegmentationViewer(pg.GraphicsLayoutWidget):
    def __init__(self, image, masks):
        """
        image: 2D numpy array (Nx, Ny) - the background image
        masks: list (or array) of 2D masks (shape: (N_classes, Nx, Ny))
        """
        super().__init__()
        self.image = image
        self.masks = masks

        # Track which masks are accepted
        self.accepted_masks = set()
        # Keep a history (stack) to allow undo
        self.accepted_stack = []

        # Define tab10 colormap as a list of RGB tuples
        self.tab10_colors = [
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # blue
            (1.0, 0.4980392156862745, 0.054901960784313725),                # orange
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # green
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # red
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),    # purple
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # brown
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),   # pink
            (0.0, 0.5, 0.5),   # teal
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # olive
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # cyan
        ]

        # Create a 2x1 layout:
        #  - Left view: initially shows all segmentations
        #  - Right view: initially shows none (empty)
        self.left_view = self.addViewBox(row=0, col=0)
        self.right_view = self.addViewBox(row=0, col=1)

        # Keep the same aspect ratio in both
        self.left_view.setAspectLocked(True)
        self.right_view.setAspectLocked(True)

        # Base images (one for the left, one for the right)
        self.left_base_img_item = pg.ImageItem(self.image)
        self.right_base_img_item = pg.ImageItem(self.image)
        self.left_view.addItem(self.left_base_img_item)
        self.right_view.addItem(self.right_base_img_item)

        # For convenience, create parallel lists of mask items:
        #   left_mask_items[i]: the overlay for mask i in the left view
        #   right_mask_items[i]: the overlay for mask i in the right view
        self.left_mask_items = []
        self.right_mask_items = []

        # Connect mouse events in the left scene
        # (We'll assume we only accept clicks in the left scene,
        # but you could also handle right scene events if needed.)
        self.scene().sigMouseClicked.connect(self.mouse_clicked)        

        # Make sure we can receive key events
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()       


    def initialize_overlays(self):
        """
        Create overlays for all masks.
        """
        for i, mask in enumerate(self.masks):
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
        
        
    def create_overlay_rgba(self, mask, index = 0):
        """
        Convert a single 2D mask (Nx x Ny) into an RGBA overlay using the tab10 colormap.
        """
        Nx, Ny = mask.shape
        rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)

        # Get the color corresponding to the index
        color = self.tab10_colors[index % len(self.tab10_colors)]

        # Apply the color to the mask
        inds = mask > 0.5
        rgba[inds, 0] = color[0]  # Red channel
        rgba[inds, 1] = color[1]  # Green channel
        rgba[inds, 2] = color[2]  # Blue channel
        rgba[inds, 3] = 1.0       # Alpha channel for transparency

        return rgba
    
    def load_data(self, base_image, masks, class_dict = None):
        """
        Load new base image and masks into the viewer.
        """
        
        self.base_image = base_image
        self.masks = masks

        # Reset accepted masks and history
        if len(class_dict) > 1:
            # Clear indices for all classes in the class dictionary
            for class_name in self.class_dict.keys():
                # Clear all mask indices for this class
                self.class_dict[class_name]['masks'].clear()     
        else:
            self.accepted_masks.clear()
            self.accepted_stack.clear()        

        # Update the base images
        self.left_base_img_item.setImage(self.base_image)
        self.right_base_img_item.setImage(self.base_image)

        # Clear old masks
        for item in self.left_mask_items:
            self.left_view.removeItem(item)
        for item in self.right_mask_items:
            self.right_view.removeItem(item)

        # Reload masks
        self.left_mask_items.clear()
        self.right_mask_items.clear()

        self.initialize_overlays()    

    def mouse_clicked(self, event):
        """
        Only accept clicks in the left view => i.e., the left scene items.
        We'll do pixel-based picking by mapping scene coords to the left image.
        """

        # Where did we click in scene coordinates?
        scene_pos = event.scenePos()

        # Convert that position to local coords in the LEFT base image
        # so that x,y correspond to array indices for 'self.image'.
        image_pos = self.left_base_img_item.mapFromScene(scene_pos)
        x, y = int(image_pos.x()), int(image_pos.y())

        Nx, Ny = self.image.shape[:2]
        if not (0 <= x < Nx and 0 <= y < Ny):
            print(f"Clicked out of bounds: {x}, {y}")
            return  # clicked out of bounds        

        # Check which segmentation(s) cover that pixel in the left image
        mask_hits = []
        for i in range(len(self.masks)):
            if self.masks[i][x, y] > 0:
                mask_hits.append(i)

        # If this is the first click at this position, store the mask hits
        if not hasattr(self, '_last_click_pos') or self._last_click_pos != (x, y):
            self._last_click_pos = (x, y)
            self._current_mask_index = 0
        else:
            # Cycle to the next mask
            self._current_mask_index = (self._current_mask_index + 1) % len(mask_hits)

        # Get the current mask to select
        i_hit = mask_hits[self._current_mask_index]

        # If it's a left-click, we "accept" => hide left, show right
        if event.button() == QtCore.Qt.LeftButton:
            # Only do something if it's currently visible on the left
            if self.left_mask_items[i_hit].isVisible():
                self.left_mask_items[i_hit].setVisible(False)
                self.right_mask_items[i_hit].setVisible(True)

                # Add to accepted set
                self.accepted_masks.add(i_hit)
                # Push onto the stack for undo
                self.accepted_stack.append(i_hit)

                print(self.accepted_masks)
                print(f"Accepted mask {i_hit}: now hidden on left, shown on right.")
        print(f"Current accepted masks: {self.accepted_masks}")


    def keyPressEvent(self, event):
        """
        Press 'r' => undo the last accepted mask:
            hide it on the right, show it on the left
        """
        key = event.key()
        if key == QtCore.Qt.Key_R:
            # Undo => pop from stack, remove from accepted_masks, hide on right, show on left
            if self.accepted_stack:
                i = self.accepted_stack.pop()
                if i in self.accepted_masks:
                    self.accepted_masks.remove(i)
                # Hide it on the right, show on the left
                self.right_mask_items[i].setVisible(False)
                self.left_mask_items[i].setVisible(True)
                print(f"Undid mask {i}: hidden on right, shown on left.")
            else:
                print("No mask to undo!")
        else:
            super().keyPressEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Create a dummy base image
    Nx, Ny = 256, 256
    base_image = np.random.normal(loc=128, scale=20, size=(Nx, Ny))

    # Create some dummy masks
    mask1 = np.zeros((Nx, Ny))
    mask1[50:100, 50:100] = 1
    mask2 = np.zeros((Nx, Ny))
    mask2[80:130, 100:150] = 1
    mask3 = np.zeros((Nx, Ny))
    mask3[150:200, 30:80] = 1
    masks = [mask1, mask2, mask3]

    window = SegmentationViewer(base_image, masks)
    window.resize(1100, 600)  # Set default size: width=1200, height=800    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

