from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QListWidget, QPlainTextEdit,
    QVBoxLayout, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
from saber.gui.text.segmentation_viewer import HashtagSegmentationViewer
from saber.gui.text.hashtag_manager import HashtagManager
from saber.gui.text.text_annotation import (
    GlobalDescriptionWidget, SegmentationDescriptionWidget, 
    ControlPanelWidget, HashtagListWidget
)
import sys, zarr, click, json, os
from typing import List
import numpy as np
from datetime import datetime

class TextAnnotationWindow(QMainWindow):
    """Main window for per-segmentation text annotation with hashtag organization."""
    
    def __init__(self, zarr_path: str, save_path: str):
        super().__init__()
        
        # Initialize core data
        self.zarr_path = zarr_path
        self.save_path = save_path
        self.load_zarr_data()
        
        # Initialize text storage
        self.global_descriptions = {}  # run_id -> global description text
        self.segmentation_descriptions = {}  # run_id -> {segmentation_id -> description}
        
        # Initialize hashtag manager
        self.hashtag_manager = HashtagManager()
        
        # Load existing data
        self.load_existing_text_data()
        
        # Setup UI
        self.setup_ui()
        self.setup_segmentation_viewer()
        self.setup_connections()
        
        # Setup update timer
        self.setup_hashtag_timer()
        
        # Load initial data
        self.load_text_for_current_image()

        # Track currently selected segmentation
        self.current_selected_id = None  

    def load_zarr_data(self):
        """Load zarr data and run IDs."""
        if os.path.exists(self.zarr_path):
            self.root = zarr.open(self.zarr_path, mode='r')
        else:
            raise FileNotFoundError(f"Zarr file {self.zarr_path} does not exist.")
        self.run_ids = list(self.root.keys())
        self.good_run_ids = []

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
        self.main_splitter.setSizes([125, 1000, 100])
        self.resize(1225, 750)

    def setup_left_panel(self):
        """Setup the left panel with run ID list."""
        self.left_panel = QWidget()
        layout = QVBoxLayout(self.left_panel)
        
        self.image_list = QListWidget()
        for image_name in self.run_ids:
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
        # Read initial data
        initial_image, initial_masks = self.read_data(self.run_ids[0])
        
        # Create enhanced segmentation viewer
        self.segmentation_viewer = HashtagSegmentationViewer(initial_image, initial_masks)
        self.segmentation_viewer.initialize_overlays()

        # Set up callbacks for segmentation selection
        self.segmentation_viewer.set_selection_callbacks(
            selection_callback=self.on_segmentation_selected_from_viewer,
            deselection_callback=self.on_segmentation_deselected_from_viewer
        )

        # Make the viewer 40% taller than default
        default_width = 1100
        default_height = 600
        new_height = int(default_height * 1.4)  # Fixed height calculation
        
        self.segmentation_viewer.resize(default_width, new_height)
        
        # Insert into middle panel layout
        self.middle_layout.insertWidget(1, self.segmentation_viewer)
        self.middle_layout.setStretchFactor(self.segmentation_viewer, 1)

    def setup_connections(self):
        """Setup signal connections."""
        # List selection
        self.image_list.itemClicked.connect(self.on_image_selected)
        
        # Text changes
        self.global_desc_widget.textChanged.connect(self.on_text_changed)
        self.seg_desc_widget.textChanged.connect(self.on_text_changed)
        
        # Control panel
        self.control_panel.saveClicked.connect(self.save_segmentation)

    def setup_hashtag_timer(self):
        """Simplified timer setup - only color updates needed."""
        self.color_update_timer = QTimer()
        self.color_update_timer.setSingleShot(True)
        self.color_update_timer.timeout.connect(self.update_segmentation_colors)

    # Event handlers
    def on_image_selected(self, item):
        """Clean run switching - immediate cache clear and fresh start."""
        run_id = item.text()
        
        # Skip if same run ID
        if hasattr(self, '_current_run_id') and self._current_run_id == run_id:
            return
        
        print(f"Switching to run: {run_id}")
        
        # 1. IMMEDIATE CACHE CLEAR - assume all data was saved
        self.clear_all_ui_state()
        
        # 2. Load new data
        self._current_run_id = run_id
        base_image, masks = self.read_data(run_id)
        
        # 3. Fresh viewer state
        self.segmentation_viewer.load_data_fresh(base_image, masks)
        
        # 4. Load text data for new run (keep hashtags, clear UI)
        self.load_fresh_text_data(run_id)
        
        print(f"Successfully switched to {run_id}")

    def clear_all_ui_state(self):
        """Immediately clear all UI state - assume data was saved."""
        # Clear text widgets (don't save, assume it was already saved)
        self.global_desc_widget.textChanged.disconnect()
        self.seg_desc_widget.textChanged.disconnect()
        
        try:
            # Clear text fields
            self.global_desc_widget.set_text("")
            self.seg_desc_widget.clear_selection()
            
            # Clear selection state
            self.current_selected_id = None
            
            # Clear viewer highlights
            self.segmentation_viewer.clear_highlight()
            
        finally:
            # Reconnect after clearing
            self.global_desc_widget.textChanged.connect(self.on_text_changed)
            self.seg_desc_widget.textChanged.connect(self.on_text_changed)

    def load_fresh_text_data(self, run_id: str):
        """Load text data for new run - keep hashtag colors, update content."""
        # Disconnect to prevent cascading updates during load
        self.global_desc_widget.textChanged.disconnect()
        self.seg_desc_widget.textChanged.disconnect()
        
        try:
            # Load global description for this run
            global_text = self.global_descriptions.get(run_id, "")
            self.global_desc_widget.set_text(global_text)
            
            # Update hashtags for this run only
            self.update_hashtags_for_run(run_id)
            
            # Update colors based on existing descriptions for this run
            self.update_colors_for_run(run_id)
            
        finally:
            # Reconnect signals
            self.global_desc_widget.textChanged.connect(self.on_text_changed)
            self.seg_desc_widget.textChanged.connect(self.on_text_changed)

    def update_hashtags_for_run(self, run_id: str):
        """Update hashtags for specific run only."""
        # Clear hashtags for this run only
        self.hashtag_manager.clear_run_hashtags(run_id)
        
        # Add hashtags from global description
        global_text = self.global_descriptions.get(run_id, "")
        if global_text:
            self.hashtag_manager.add_hashtags_from_global(run_id, global_text)
        
        # Add hashtags from segmentation descriptions
        if run_id in self.segmentation_descriptions:
            for seg_id, seg_text in self.segmentation_descriptions[run_id].items():
                self.hashtag_manager.add_hashtags_from_segmentation(run_id, seg_id, seg_text)
        
        # Update hashtag UI
        self.hashtag_manager.update_hashtag_list_widget(self.hashtag_widget.get_list_widget())

    def update_colors_for_run(self, run_id: str):
        """Update segmentation colors for specific run."""
        if run_id not in self.segmentation_descriptions:
            return
        
        color_mapping = {}
        for seg_id, description in self.segmentation_descriptions[run_id].items():
            hashtags = self.hashtag_manager.extract_hashtags(description)
            if hashtags:
                first_hashtag = sorted(hashtags)[0]
                color_mapping[int(seg_id)] = self.hashtag_manager.get_hashtag_color(first_hashtag)
        
        self.segmentation_viewer.update_mask_colors(color_mapping)

    def on_segmentation_selected_from_viewer(self, segmentation_id: int):
        """Handle segmentation selection from the viewer."""
        self.select_segmentation(segmentation_id)

    def on_segmentation_deselected_from_viewer(self):
        """Handle segmentation deselection from the viewer."""
        # Save current text before deselecting
        self.save_text_to_memory()
        
        self.seg_desc_widget.clear_selection()
        self.clear_selection_highlight()
        self.current_selected_id = None
        
        # Update hashtags when deselecting to capture any final changes
        current_run_id = self.get_current_run_id()
        if current_run_id:
            self.update_hashtags_for_run(current_run_id)

    def select_segmentation(self, segmentation_id: int):
        """Simple segmentation selection - no complex state management."""
        # Save current text quickly
        self.save_text_to_memory()
        
        # Clear previous selection
        self.clear_selection_highlight()
        
        # Set new selection
        self.seg_desc_widget.set_selected_segmentation(segmentation_id)
        self.current_selected_id = segmentation_id
        
        # Highlight
        self.add_selection_highlight(segmentation_id)
        
        # Load description
        current_run_id = self.get_current_run_id()
        if (current_run_id in self.segmentation_descriptions and 
            str(segmentation_id) in self.segmentation_descriptions[current_run_id]):
            existing_text = self.segmentation_descriptions[current_run_id][str(segmentation_id)]
            self.seg_desc_widget.set_text(existing_text)
        else:
            self.seg_desc_widget.set_text("")
        
        # Update hashtags for current run
        self.update_hashtags_for_run(current_run_id)

    def on_text_changed(self):
        """Simple text change handler - just save and update colors."""
        self.save_text_to_memory()
        
        # Update colors with short delay
        self.color_update_timer.start(200)

    def add_selection_highlight(self, segmentation_id: int):
        """Add a boundary highlight around the selected segmentation."""
        self.segmentation_viewer.highlight_mask(segmentation_id)

    def clear_selection_highlight(self):
        """Clear any existing selection highlight."""
        self.segmentation_viewer.clear_highlight()

    def update_segmentation_colors(self):
        """Simple color update for current run."""
        current_run_id = self.get_current_run_id()
        if current_run_id:
            self.update_colors_for_run(current_run_id)

    # Data management methods
    def get_current_run_id(self) -> str:
        """Get the currently selected run ID."""
        current_row = self.image_list.currentRow()
        return self.run_ids[current_row] if current_row >= 0 else None

    def read_data(self, run_id: str):
        """Read image and mask data for a run ID."""
        base_image = self.root[run_id]['image'][:]
        masks = self.root[run_id]['masks'][:]

        (nx, ny) = base_image.shape
        if nx < ny:
            base_image = base_image.T
            masks = np.swapaxes(masks, 1, 2)

        return base_image, masks

    def save_text_to_memory(self):
        """Save current text to memory."""
        current_run_id = self.get_current_run_id()
        if not current_run_id:
            return
        
        # Save global description
        global_text = self.global_desc_widget.get_text().strip()
        if global_text:
            self.global_descriptions[current_run_id] = global_text
        elif current_run_id in self.global_descriptions:
            del self.global_descriptions[current_run_id]
        
        # Save segmentation description
        selected_id = self.seg_desc_widget.get_selected_id()
        if selected_id is not None:
            seg_text = self.seg_desc_widget.get_text().strip()
            if current_run_id not in self.segmentation_descriptions:
                self.segmentation_descriptions[current_run_id] = {}
            
            seg_key = str(selected_id)
            if seg_text:
                self.segmentation_descriptions[current_run_id][seg_key] = seg_text
            elif seg_key in self.segmentation_descriptions[current_run_id]:
                del self.segmentation_descriptions[current_run_id][seg_key]
            
            # Clean up empty entries
            if not self.segmentation_descriptions[current_run_id]:
                del self.segmentation_descriptions[current_run_id]

    def load_text_for_current_image(self):
        """Simple initial load - just load text and update hashtags."""
        current_run_id = self.get_current_run_id()
        if not current_run_id:
            return
        
        self.load_fresh_text_data(current_run_id)

    def load_existing_text_data(self):
        """Load existing text data from save file."""
        if self.save_path and os.path.exists(self.save_path):
            try:
                save_root = zarr.open(self.save_path, mode='r')
                if 'text_annotations' in save_root.attrs:
                    text_data = json.loads(save_root.attrs['text_annotations'])
                    self.global_descriptions = text_data.get('global_descriptions', {})
                    self.segmentation_descriptions = text_data.get('segmentation_descriptions', {})
                    self.hashtag_manager.load_data_from_save(text_data)
                    print("Loaded existing text annotations.")
            except Exception as e:
                print(f"Could not load existing text data: {e}")

    # Action methods
    def save_segmentation(self):
        """Save segmentation masks and text data."""
        if self.save_path is None:
            print("\nCurrently in viewer mode.\nSave path is not set.")
            return
        
        # Save text data first
        self.save_text_to_memory()
        self.save_text_data()
        
        # Save segmentation data
        self.save_masks_data()

    def save_text_data(self):
        """Save text data to zarr file."""
        if not self.save_path:
            QMessageBox.warning(self, "Warning", "No save path specified.")
            return
        
        try:
            zarr_root = zarr.open(self.save_path, mode='a')
            text_data = {
                'global_descriptions': self.global_descriptions,
                'segmentation_descriptions': self.segmentation_descriptions,
                **self.hashtag_manager.get_data_for_save()
            }
            zarr_root.attrs['text_annotations'] = json.dumps(text_data)
            print("Text annotations saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save text data: {e}")

    def save_masks_data(self):
        """Save mask data to zarr file."""
        zarr_root = zarr.open(self.save_path, mode='a')
        
        # Save current run data
        current_row = self.image_list.currentRow()
        run_id = self.run_ids[current_row]
        
        if run_id in zarr_root:
            print(f"\nWarning: Overwriting existing group {run_id}")
        
        segmentation_group = zarr_root.require_group(run_id)
        current_image = self.segmentation_viewer.left_base_img_item.image
        segmentation_group['image'] = current_image
        
        try:
            self.save_masks_to_zarr(segmentation_group, run_id)
            zarr_root.attrs['good_run_ids'] = self.good_run_ids
        except Exception as e:
            print(f"Error saving masks for run ID {run_id}: {e}")

    def save_masks_to_zarr(self, segmentation_group, run_id):
        """Save masks with integrated descriptions as structured data."""
        total_masks = len(self.segmentation_viewer.masks)
        
        if total_masks == 0:
            print(f"No masks to save for run ID {run_id}")
            return
        
        # Get accepted masks
        accepted_indices = getattr(self.segmentation_viewer, 'accepted_masks', set())
        
        if not accepted_indices:
            print(f"No accepted masks to save for run ID '{run_id}'.")
            return
        
        # Save accepted masks with descriptions
        masks_group = segmentation_group.require_group('masks')
        
        for i in accepted_indices:
            if 0 <= i < total_masks:
                seg_group = masks_group.require_group(f'segmentation_{i}')
                
                # Store mask array
                seg_group['mask'] = self.segmentation_viewer.masks[i].astype(np.uint8)
                
                # Store description and metadata as attributes
                description = self.segmentation_descriptions.get(run_id, {}).get(str(i), '')
                hashtags = list(self.hashtag_manager.extract_hashtags(description))
                bbox = self._get_mask_bbox(self.segmentation_viewer.masks[i])
                
                seg_group.attrs['description'] = description
                seg_group.attrs['hashtags'] = json.dumps(hashtags)
                seg_group.attrs['bbox'] = bbox
                seg_group.attrs['area'] = int(np.sum(self.segmentation_viewer.masks[i] > 0))
                seg_group.attrs['segmentation_id'] = i
        
        print(f'Saved {len(accepted_indices)} mask+description pairs for runID: {run_id}')
        
        # Save rejected masks (simple, no descriptions)
        all_indices = set(range(total_masks))
        rejected_indices = all_indices - accepted_indices
        
        if rejected_indices:
            rejected_group = segmentation_group.require_group('rejected_masks')
            for idx in rejected_indices:
                if 0 <= idx < total_masks:
                    rejected_group[f'rejected_{idx}'] = self.segmentation_viewer.masks[idx].astype(np.uint8)

    def _get_mask_bbox(self, mask: np.ndarray):
        """Get bounding box for a mask."""
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return [0, 0, 0, 0]
        
        return [
            int(cols.min()), int(rows.min()),  # x_min, y_min
            int(cols.max()), int(rows.max())   # x_max, y_max
        ]

    def load_masks_with_descriptions(self, run_id: str):
        """Load masks with their descriptions as a unified dictionary."""
        if not os.path.exists(self.save_path):
            return {}
        
        try:
            zarr_root = zarr.open(self.save_path, mode='r')
            if run_id not in zarr_root or 'masks' not in zarr_root[run_id]:
                return {}
            
            masks_group = zarr_root[run_id]['masks']
            result = {}
            
            for seg_key in masks_group.keys():
                seg_group = masks_group[seg_key]
                if 'mask' in seg_group:
                    result[seg_key] = {
                        'mask': seg_group['mask'][:],
                        'description': seg_group.attrs.get('description', ''),
                        'hashtags': json.loads(seg_group.attrs.get('hashtags', '[]')),
                        'bbox': list(seg_group.attrs.get('bbox', [0, 0, 0, 0])),
                        'area': seg_group.attrs.get('area', 0),
                        'segmentation_id': seg_group.attrs.get('segmentation_id', 0)
                    }
            
            return result
            
        except Exception as e:
            print(f"Error loading masks with descriptions for {run_id}: {e}")
            return {}

    # Navigation methods
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        current_row = self.image_list.currentRow()

        if event.key() == Qt.Key_Left:
            self.load_next_runID(current_row - 1)
        elif event.key() == Qt.Key_Right:
            self.load_next_runID(current_row + 1)
        elif event.key() == Qt.Key_S:
            self.save_segmentation()
        else:
            super().keyPressEvent(event)

    def load_next_runID(self, new_row):
        """Load the next/previous run ID."""
        new_row = max(0, min(new_row, self.image_list.count() - 1))
        self.image_list.setCurrentRow(new_row)
        self.on_image_selected(self.image_list.item(new_row))

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
@click.option('--output', type=str, required=False, default=None, 
              help="Path to the Saving Zarr file.")
def text_gui(input: str, output: str):
    """GUI for Annotating Individual SAM2 Segmentations with Hashtag-Based Text Descriptions."""

    app = QApplication(sys.argv)
    main_window = TextAnnotationWindow(input, output)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    text_gui()