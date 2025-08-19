"""
Main window for SAM2-ET text annotation GUI.
Handles UI setup and coordinates between components.
"""

import sys
import click
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QListWidget, QPlainTextEdit,
    QVBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt

from saber.gui.text.segmentation_viewer import HashtagSegmentationViewer
from saber.gui.text.hashtag_manager import HashtagManager
from saber.gui.text.text_annotation import (
    GlobalDescriptionWidget, SegmentationDescriptionWidget, 
    ControlPanelWidget, HashtagListWidget
)

from saber.gui.text.data_manager import TextAnnotationDataManager
from saber.gui.text.annotation_controller import TextAnnotationController


class TextAnnotationWindow(QMainWindow):
    """Main window for per-segmentation text annotation with hashtag organization."""
    
    def __init__(self, zarr_path: str, save_path: str):
        super().__init__()
        
        # Initialize core components
        self.data_manager = TextAnnotationDataManager(zarr_path, save_path)
        self.hashtag_manager = HashtagManager()
        self.controller = TextAnnotationController(self.data_manager, self.hashtag_manager)
        
        # Debug zarr contents
        # self.data_manager.debug_zarr_contents()
        
        # Setup UI
        self.setup_ui()
        self.setup_segmentation_viewer()
        self.setup_connections()
        
        # Load initial data for the first run
        self.load_text_for_current_image()

    def setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("SAM2-ET Tomogram Inspection GUI with Per-Segmentation Text Annotations")
        self.setup_menu_bar()
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(self.main_splitter)
        
        # Setup panels
        self.setup_left_panel()
        self.setup_middle_panel() 
        self.setup_right_panel()
        
        # Set sizes and window size (more space for images)
        self.main_splitter.setSizes([125, 1000, 150])
        self.resize(1225, 750)

    def setup_left_panel(self):
        """Setup the left panel with run ID list."""
        self.left_panel = QWidget()
        layout = QVBoxLayout(self.left_panel)
        
        self.image_list = QListWidget()
        for image_name in self.data_manager.run_ids:
            self.image_list.addItem(image_name)
        
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
        
        layout.addWidget(self.image_list)
        self.main_splitter.addWidget(self.left_panel)

    def setup_middle_panel(self):
        """Setup the middle panel with segmentation viewer and text inputs."""
        self.middle_panel = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_panel)
        
        # Reduce margins and spacing for more compact layout
        self.middle_layout.setContentsMargins(5, 5, 5, 5)
        self.middle_layout.setSpacing(2)
        
        # Global description widget
        self.global_desc_widget = GlobalDescriptionWidget()
        self.middle_layout.addWidget(self.global_desc_widget)
        
        # Control panel (save button)
        self.control_panel = ControlPanelWidget()
        self.middle_layout.addWidget(self.control_panel)
        
        # Segmentation description widget
        self.seg_desc_widget = SegmentationDescriptionWidget()
        self.middle_layout.addWidget(self.seg_desc_widget)
        
        # Set stretch factors: 0 for fixed-size widgets, 1 for the viewer
        self.middle_layout.setStretchFactor(self.global_desc_widget, 0)
        self.middle_layout.setStretchFactor(self.control_panel, 0)
        self.middle_layout.setStretchFactor(self.seg_desc_widget, 0)
        
        self.main_splitter.addWidget(self.middle_panel)

    def setup_right_panel(self):
        """Setup the right panel with hashtag list."""
        self.hashtag_widget = HashtagListWidget()
        self.main_splitter.addWidget(self.hashtag_widget)

    def setup_segmentation_viewer(self):
        """Initialize and setup the segmentation viewer."""
        # Use session-or-saved augmented loader for the first run as well
        first_run = self.data_manager.run_ids[0]
        initial_image, initial_masks, accepted = self.data_manager.read_with_session_fallback(first_run)

        self.segmentation_viewer = HashtagSegmentationViewer(initial_image, initial_masks)
        self.segmentation_viewer.initialize_overlays()
        self.segmentation_viewer.set_accepted_indices(accepted)

        # Make the viewer 40% taller than default
        default_width = 1100
        default_height = 600
        new_height = int(default_height * 1.4)
        self.segmentation_viewer.resize(default_width, new_height)

        # Insert into middle panel layout
        self.middle_layout.insertWidget(1, self.segmentation_viewer)
        self.middle_layout.setStretchFactor(self.segmentation_viewer, 1)

    def setup_connections(self):
        """Setup signal connections."""
        # Set UI components in controller
        self.controller.set_ui_components(
            segmentation_viewer=self.segmentation_viewer,
            global_desc_widget=self.global_desc_widget,
            seg_desc_widget=self.seg_desc_widget,
            hashtag_widget=self.hashtag_widget,
            image_list=self.image_list
        )
        
        # Setup controller connections
        self.controller.setup_connections()
        
        # Control panel
        self.control_panel.saveClicked.connect(self.save_segmentation)

        # When a new mask is drawn, let the controller create/init an entry for it
        self.segmentation_viewer.maskAdded.connect(self.controller.on_mask_added)

    def load_text_for_current_image(self):
        """Load text and annotations for the currently selected run."""
        current_run_id = self.controller.get_current_run_id()
        if not current_run_id:
            return
        
        print(f"🚀 Initial load for run: {current_run_id}")
        self.controller.load_run_annotations(current_run_id)

    def save_segmentation(self):
        """Save segmentation masks and text data."""
        success = self.controller.save_segmentation(self.data_manager.save_path)
        
        if success:
            current_run_id = self.controller.get_current_run_id()
            accepted_count = len(getattr(self.segmentation_viewer, 'accepted_masks', set()))
            annotated_count = len(self.data_manager.segmentation_descriptions.get(current_run_id, {}))
            print(f"✓ Saved {current_run_id}: {accepted_count} accepted masks, {annotated_count} with descriptions")
        else:
            QMessageBox.critical(self, "Error", "Failed to save data. Check console for details.")

    # Navigation methods
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        current_row = self.image_list.currentRow()

        if event.key() == Qt.Key_Left:
            self.controller.load_next_runID(current_row - 1)
        elif event.key() == Qt.Key_Right:
            self.controller.load_next_runID(current_row + 1)
        elif event.key() == Qt.Key_S:
            self.save_segmentation()
        # else:
        #     super().keyPressEvent(event)

    # UI setup helpers
    def setup_menu_bar(self):
        """Setup menu bar with help menu."""
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        welcome_action = help_menu.addAction("Show Welcome Message")
        welcome_action.triggered.connect(self.show_welcome_message)

    def show_welcome_message(self):
        """Display welcome message with instructions."""
        message = (
            "Welcome to the SAM2-ET Per-Segmentation Text Annotation GUI!\n\n"
            "Quick Tutorial:\n"
            "1. **Navigating Images**: Use Left/Right Arrow Keys\n"
            "2. **Per-Segmentation Descriptions**: Click on masks to select them\n"
            "3. **Hashtag Organization**: Use #hashtags in descriptions\n"
            "4. **Saving**: Press 'S' to save all data\n\n"
            "The hashtag panel on the right automatically organizes your annotations.\n"
            "Start by clicking on segmentation masks and adding #hashtag descriptions!"
        )

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Welcome!")
        msg_box.setIcon(QMessageBox.Information)

        text_edit = QPlainTextEdit(message)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("QPlainTextEdit { border: none; padding: 0px; background: transparent; }")
        text_edit.setFixedSize(550, 350)

        layout = msg_box.layout()
        layout.addWidget(text_edit, 0, 1, 1, layout.columnCount())
        layout.setContentsMargins(10, 10, 10, 10)

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


@click.command(context_settings={"show_default": True})
@click.option('--input', type=str, required=True, 
              help="Path to the Reading Zarr file.")
@click.option('--output', type=str, required=False, default='labels.zarr', 
              help="Path to the Saving Zarr file.")
def text_gui(input: str, output: str):
    """GUI for Annotating Individual SAM2 Segmentations with Hashtag-Based Text Descriptions."""

    app = QApplication(sys.argv)
    main_window = TextAnnotationWindow(input, output)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    text_gui()