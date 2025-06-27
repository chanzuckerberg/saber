from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QListWidget, QPlainTextEdit,
    QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt
from saber.gui.multi_class_segmentation_picker import MultiClassSegmentationViewer
from saber.gui.segmentation_picker import SegmentationViewer
import sys, zarr, click, json, os
from typing import List
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self, 
                 zarr_path: str,
                 save_path: str,
                 class_names: List[str]):
        super().__init__()

         # Display welcome message with tutorial
        self.setup_menu_bar()

        # Load Initial Segmentation Data
        if os.path.exists(zarr_path):
            self.root = zarr.open(zarr_path, mode='r')
        else:
            raise FileNotFoundError(f"Zarr file {zarr_path} does not exist.")
        self.run_ids = list(self.root.keys())

        self.save_path = save_path
        
        self.class_dict = {
            class_name: {
                'value': i + 1,  # Unique integer value for the class
                'masks': [],     # List to store masks for this class
            }
            for i, class_name in enumerate(class_names)
        }
        self.selected_class = class_names[0]

        self.setWindowTitle("SAM2-ET Tomogram Inspection GUI")
        self.good_run_ids = []

        # Create the splitter for left (list) and right (viewer)
        self.splitter = QSplitter(Qt.Horizontal, self)

        # --- Left Panel: RunIDs List ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        self.image_list = QListWidget()
        for image_name in self.run_ids:
            self.image_list.addItem(image_name)

        # Highlight the first entry by default
        if self.image_list.count() > 0:  # Ensure the list is not empty
            self.image_list.setCurrentRow(0)

        self.left_layout.addWidget(self.image_list)
        self.splitter.addWidget(self.left_panel)

        # --- Right Panel: Segmentation Viewer ---
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Read data for the first run
        (initial_image, initial_masks) = self.read_data(self.run_ids[0])
        if len(class_names) > 1:
            self.segmentation_viewer = MultiClassSegmentationViewer(
                initial_image, initial_masks, 
                self.class_dict, self.selected_class )
        else:
            self.segmentation_viewer = SegmentationViewer(
                initial_image, initial_masks)
            self.segmentation_viewer.initialize_overlays()
        self.right_layout.addWidget(self.segmentation_viewer)

        # Save Button and Dropdown at the Bottom
        bottom_layout = QHBoxLayout()

        # "Select a class" Label and Dropdown
        self.class_dropdown = QComboBox()
        self.class_dropdown.addItems(class_names)  # Add class options
        bottom_layout.addWidget(self.class_dropdown)

        # Save Button
        self.save_button = QPushButton("Save Segmentation")
        bottom_layout.addWidget(self.save_button)

        # Connect dropdown to monitor class selection
        self.class_dropdown.currentTextChanged.connect(self.check_selected_class)        

        # Add Bottom Layout to the Right Panel
        self.right_layout.addLayout(bottom_layout)
        self.splitter.addWidget(self.right_panel)

        # Set splitter as the central widget of the main window
        self.setCentralWidget(self.splitter)

        # Initial size
        self.splitter.setSizes([175, 850])  # Set left panel to 150px and right panel to 850px        
        self.resize(1000, 600)

        # Connect signals
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.save_button.clicked.connect(self.save_segmentation)  

    def check_selected_class(self, class_name):
        """
        Check the currently selected class in the dropdown menu.
        """
        self.selected_class = class_name
        self.segmentation_viewer.selected_class = class_name  # Pass to viewer
        # print(f"Currently selected class: {self.selected_class}")   

    def read_data(self, run_id):
        """
        Read the base image (tomogram) and segmentation masks for a given run ID.
        :param run_id: The ID of the selected run
        :return: base_image (numpy array), masks (list of numpy arrays)
        """

        # Get Run
        base_image = self.root[run_id]['image'][:]
        masks = self.root[run_id]['masks'][:]

        (nx, ny) = base_image.shape
        if nx < ny:
            base_image = base_image.T
            masks = np.swapaxes(masks, 1, 2)

        return base_image, masks

    def on_image_selected(self, item):
        """
        Load the selected image into the viewer.
        :param item: The selected QListWidgetItem
        """
        run_id = item.text()  # Get the selected run ID

        # Read the data for the selected run ID
        try:
            base_image, masks = self.read_data(run_id)
        except Exception as e:
            print(f"Error loading data for run ID {run_id}: {e}")
            return

        # Load the data into the segmentation viewer
        self.segmentation_viewer.load_data(base_image, masks, self.class_dict)

        # Reset the dropdown to the first class
        if len(self.class_dict) > 1:
            self.class_dropdown.setCurrentIndex(0)

    def keyPressEvent(self, event):
        """
        Handle key press events to navigate run_ids using Q and W keys.
        """
        current_row = self.image_list.currentRow()

        if event.key() == Qt.Key_Left:  # Left key for previous run_id
            new_row = current_row - 1
            self.load_next_runID(new_row)
        elif event.key() == Qt.Key_Right:  # Right key for next run_id
            new_row = current_row + 1
            self.load_next_runID(new_row)
        elif event.key() == Qt.Key_Up:
            current_index = self.class_dropdown.currentIndex()
            if current_index > 0:  # Ensure it's not the first item
                self.class_dropdown.setCurrentIndex(current_index - 1)
        elif event.key() == Qt.Key_Down:
            current_index = self.class_dropdown.currentIndex()
            if current_index < self.class_dropdown.count() - 1:  # Ensure it's not the last item
                self.class_dropdown.setCurrentIndex(current_index + 1)
        elif event.key() == Qt.Key_D:   
            if self.run_ids[current_row] not in self.good_run_ids:            
                self.good_run_ids.append(self.run_ids[current_row])
                print(f'\nSaving Current RunID: {self.run_ids[current_row]}')
                print('Current List of Good RunIDs: ', self.good_run_ids)
        elif event.key() == Qt.Key_F:
            if self.run_ids[current_row] in self.good_run_ids:            
                self.good_run_ids.remove(self.run_ids[current_row])
                print(f'\nRemoving Current RunID: {self.run_ids[current_row]}')
                print('Current List of Good RunIDs: ', self.good_run_ids)
        elif event.key() == Qt.Key_S:  # S key to save segmentation
            self.save_segmentation()            
        else:
            # If it's not Q or W, pass to the base class
            super().keyPressEvent(event)
            return           

    def load_next_runID(self, new_row):
        """
        Triggered when a new image is selected from the list.
        """
        # Ensure the new row is within bounds
        new_row = max(0, min(new_row, self.image_list.count() - 1))

        # Update the selection in the image list
        self.image_list.setCurrentRow(new_row)

        # Trigger the on_image_selected logic
        self.on_image_selected(self.image_list.item(new_row)) 

    def save_segmentation(self):
        """
        Save the current segmentation masks to a new Zarr file.
        """

        if self.save_path is None:
            # Issue a warning and exit the function
            print("\nCurrently in viewer mode.\nSave path is not set. Segmentation not saved.")
            return
        
        zarr_root = zarr.open(self.save_path, mode='a')

        # Save the User Class Dictionary to the Zarr file
        filtered_class_dict = {
            class_name: {k: v for k, v in class_data.items() if k != 'masks'}
            for class_name, class_data in self.class_dict.items()
        }
        zarr_root.attrs['class_names'] = json.dumps(filtered_class_dict)        

        # Reference the current run ID (from the selected item in the list)
        current_row = self.image_list.currentRow()        
        run_id = self.run_ids[current_row]

        # Check if the group already exists, warn if overwriting
        if run_id in zarr_root:
            print(f"\nWarning: Overwriting existing group {run_id} in {self.save_path}")

        # Create or open the group for the segmentation
        segmentation_group = zarr_root.require_group(run_id)

        # Save the base image
        current_image = self.segmentation_viewer.left_base_img_item.image
        segmentation_group['image'] = current_image

        # Save masks to Zarr
        try:                    self.save_masks_to_zarr(segmentation_group, run_id)
        except Exception as e:  print(f"Error saving masks for run ID {run_id}: {e}")   

        # Save the good_run_ids list as a dataset
        try:
            # Store the list of good_run_ids as a dataset in the root group
            zarr_root.attrs['good_run_ids'] = self.good_run_ids
        except Exception as e:
            print(f"Error updating 'good_run_ids' dataset: {e}")          

    def save_masks_to_zarr(self, segmentation_group, run_id):
        """
        Save masks to the Zarr file for both single-class and multi-class scenarios.
        
        :param segmentation_group: The Zarr group where masks will be saved.
        :param run_id: The current run ID for logging.
        """

        total_masks = len(self.segmentation_viewer.masks)
        mask_shape = self.segmentation_viewer.masks[0].shape
        accepted_masks = []  # List to store accepted masks in the correct order
        all_used_indices = set()  # Track all used indices

        # Save accepted masks (per class)
        # Sort class names by their associated 'value'
        sorted_classes = sorted(self.class_dict.keys(), key=lambda x: self.class_dict[x]['value'])
        for class_name in sorted_classes:
            class_merged_mask = np.zeros(mask_shape, dtype=np.uint8)
            
            # Get indices for this class (either from class_dict or accepted_masks for single class)
            if len(self.class_dict) > 1:
                class_indices = sorted(self.class_dict[class_name]['masks'])
            else:
                class_indices = sorted(self.segmentation_viewer.accepted_masks)

            for i in class_indices:
                if 0 <= i < total_masks:  # Validate index
                    class_merged_mask = np.logical_or(
                        class_merged_mask, self.segmentation_viewer.masks[i] > 0
                    ).astype(np.uint8)
                    all_used_indices.add(i)
                else:
                    print(f"Invalid mask index {i} in class '{class_name}', skipping.")

            # Append the merged mask for this class (even if it remains empty)
            accepted_masks.append(class_merged_mask)

        # Save accepted masks to the 'masks' group
        if accepted_masks:
            segmentation_group['masks'] = np.stack(accepted_masks).astype(np.uint8)
        else:
            print(f"No accepted masks to save for run ID '{run_id}'.")

        # Save rejected masks to a separate 'rejected_masks' group
        all_indices = set(range(total_masks))
        rejected_indices = all_indices - all_used_indices

        if rejected_indices:
            rejected_masks = []  # List to store rejected masks
            for i in rejected_indices:
                if 0 <= i < total_masks:  # Validate index
                    rejected_masks.append(self.segmentation_viewer.masks[i])
                else:
                    print(f"Invalid mask index {i} in rejected_masks, skipping.")

            # Save the rejected masks group if there are any
            if rejected_masks:
                segmentation_group['rejected_masks'] = np.stack(rejected_masks).astype(np.uint8)
            else:
                print(f"No valid rejected masks to save for run ID '{run_id}'.")
        else:
            segmentation_group['rejected_masks'] = np.array([])
            
        # Print the masks saved for the runID
        if accepted_masks:
            print('Masks saved for runID: ', run_id)

    def setup_menu_bar(self):
        """
        Sets up the menu bar with a 'Help' menu.
        """
        menu_bar = self.menuBar()

        # Add "Help" menu
        help_menu = menu_bar.addMenu("Help")

        # Add "Show Welcome Message" action to the Help menu
        welcome_action = help_menu.addAction("Show Welcome Message")
        welcome_action.triggered.connect(self.show_welcome_message)


    def show_welcome_message(self):
        """
        Displays a welcome message with basic instructions on how to use the GUI.
        """
    
        message = (
            "Welcome to the SAM2-ET Tomogram Inspection GUI!\n\n"
            "Quick Tutorial:\n"
            "1. **Navigating Images**:\n"
            "   - Use the Left Arrow Key to go to the previous image.\n"
            "   - Use the Right Arrow Key to go to the next image.\n\n"
            "2. **Class Selection**:\n"
            "   - Use the dropdown menu at the bottom to select a segmentation class.\n"
            "   - Alternatively, use the Up and Down Arrow Keys to cycle through classes.\n\n"
            "3. **Marking Images as Good or Bad**:\n"
            "   - Press 'D' to mark the current image as 'Good'.\n"
            "   - Press 'F' to unmark the current image.\n\n"
            "4. **Saving Segmentations**:\n"
            "   - Press 'S' to save your segmentations to the specified save path.\n\n"
            "5. **Mouse Actions:**\n"
            "   - Use the mouse to draw, erase, or edit segmentation masks.\n\n"
            "Start exploring by selecting an image from the list on the left!"
        )

        # Create the message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Welcome to the SAM2-ET Tomogram Inspection GUI!")
        msg_box.setIcon(QMessageBox.Information)

        # Add the message to a QPlainTextEdit for better control
        text_edit = QPlainTextEdit(message)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(
            "QPlainTextEdit { border: none; padding: 0px; background: transparent; }"
        )
        text_edit.setFixedSize(500, 400)  # Adjust size to reduce white space

        # Set the layout of the QMessageBox
        layout = msg_box.layout()
        layout.addWidget(text_edit, 0, 1, 1, layout.columnCount())
        layout.setContentsMargins(10, 10, 10, 10)  # Adjust margins

        # Add an OK button
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Show the message box
        msg_box.exec_()

@click.command(context_settings={"show_default": True})
@click.option('--input', type=str, required=True, 
              help="Path to the Reading Zarr file.")
@click.option('--output', type=str, required=False, default=None, 
              help="Path to the Saving Zarr file.")
@click.option('--class-names', type=str, required=False, default=None, 
              help="Comma separated list of class names if multiple classes are present. Keep empty if only one class is present.")
def gui(
    input: str,
    output: str,
    class_names: List[str]
    ):
    """
    GUI for Annotating SAM2 Segmentations for the Domain Expert Classifier.
    """
    
    # Convert the comma-separated string into a list of strings
    if class_names is not None:
        class_names = [name.strip() for name in class_names.split(",")]
    else:
        print("\nNo class names provided. Using default class name: 'object'")
        class_names = ['object']
    
    # Start the app
    app = QApplication(sys.argv)
    main_window = MainWindow(input, output, class_names)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
