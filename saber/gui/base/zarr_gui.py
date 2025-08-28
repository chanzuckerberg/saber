from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QListWidget, QPlainTextEdit,
    QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QComboBox, QMessageBox,
    QLineEdit, QListWidgetItem, QInputDialog, QColorDialog, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
import sys, zarr, click, json, os
import numpy as np
from datetime import datetime
from saber.gui.base.annotation_viewer import AnnotationSegmentationViewer


class ClassManagerWidget(QWidget):
    """Widget for managing segmentation classes dynamically"""
    
    classAdded = pyqtSignal(str, dict)
    classRemoved = pyqtSignal(str)
    classSelected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.class_dict = {}
        self.selected_class = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Segmentation Classes")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        add_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter class name...")
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_class)
        self.class_input.returnPressed.connect(self.add_class)
        
        add_layout.addWidget(self.class_input)
        add_layout.addWidget(self.add_btn)
        layout.addLayout(add_layout)
        
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self.on_class_selected)
        layout.addWidget(self.class_list)
        
        self.remove_btn = QPushButton("Remove Selected Class")
        self.remove_btn.clicked.connect(self.remove_class)
        self.remove_btn.setEnabled(False)
        layout.addWidget(self.remove_btn)
        
        self.setLayout(layout)
        self.add_default_class()
    
    def add_default_class(self):
        """Add a default 'object' class"""
        default_name = "object"
        color = self.get_next_color(0)
        self.add_class_to_dict(default_name, color)
    
    def get_next_color(self, index):
        """Get the next color from the TAB10 colormap"""
        TAB10_COLORS = [
            (31, 119, 180),   # blue
            (255, 127, 14),   # orange
            (44, 160, 44),    # green
            (214, 39, 40),    # red
            (148, 103, 189),  # purple
            (140, 86, 75),    # brown
            (227, 119, 194),  # pink
            (0, 128, 128),    # teal
            (188, 189, 34),   # olive
            (23, 190, 207),   # cyan
        ]
        
        rgb = TAB10_COLORS[index % len(TAB10_COLORS)]
        return QColor(rgb[0], rgb[1], rgb[2])
    
    def add_class(self):
        """Add a new class to the list"""
        class_name = self.class_input.text().strip()
        
        if not class_name:
            QMessageBox.warning(self, "Warning", "Please enter a class name.")
            return
            
        if class_name in self.class_dict:
            QMessageBox.warning(self, "Warning", f"Class '{class_name}' already exists.")
            return
        
        # Get the next color automatically
        color = self.get_next_color(len(self.class_dict))
        
        self.add_class_to_dict(class_name, color)
        self.class_input.clear()
    
    def add_class_to_dict(self, class_name, color):
        """Add class to internal dictionary and update UI"""
        self.class_dict[class_name] = {
            'value': len(self.class_dict) + 1,
            'color': color,
            'masks': []
        }
        
        item = QListWidgetItem(class_name)
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))
        
        self.class_list.addItem(item)
        self.class_list.setCurrentItem(item)
        self.on_class_selected(item)
        
        self.classAdded.emit(class_name, self.class_dict[class_name])
    
    def remove_class(self):
        """Remove the selected class"""
        current_item = self.class_list.currentItem()
        if not current_item:
            return
        
        class_name = current_item.text()
        
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to remove class '{class_name}'?\n"
            f"This will remove all associated mask assignments.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.class_dict[class_name]
            self.class_list.takeItem(self.class_list.row(current_item))
            
            for i, name in enumerate(self.class_dict.keys()):
                self.class_dict[name]['value'] = i + 1
            
            self.classRemoved.emit(class_name)
            
            if self.class_list.count() > 0:
                self.class_list.setCurrentRow(0)
                self.on_class_selected(self.class_list.currentItem())
            else:
                self.selected_class = None
                self.remove_btn.setEnabled(False)
    
    def on_class_selected(self, item):
        """Handle class selection"""
        if item:
            self.selected_class = item.text()
            self.remove_btn.setEnabled(True)
            self.classSelected.emit(self.selected_class)
    
    def get_selected_class(self):
        return self.selected_class
    
    def get_class_dict(self):
        return self.class_dict


class MainWindow(QMainWindow):
    def __init__(self, zarr_path: str):
        super().__init__()
        
        self.setup_menu_bar()
        
        # Load Zarr data
        if os.path.exists(zarr_path):
            self.root = zarr.open(zarr_path, mode='r')
        else:
            raise FileNotFoundError(f"Zarr file {zarr_path} does not exist.")
        self.run_ids = list(self.root.keys())
        
        # Initialize the annotations dictionary directly
        # ALL run_ids start with empty annotations
        self.annotations = {run_id: {} for run_id in self.run_ids}
        
        self.setWindowTitle("SAM2-ET Dynamic Annotation GUI")
        
        # Create the main splitter
        self.main_splitter = QSplitter(Qt.Horizontal, self)
        
        # --- Left Panel: RunIDs List ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        self.image_list = QListWidget()
        for image_name in self.run_ids:
            self.image_list.addItem(image_name)
        
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
        
        self.left_layout.addWidget(self.image_list)
        self.main_splitter.addWidget(self.left_panel)
        
        # --- Middle Panel: Segmentation Viewer ---
        self.middle_panel = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_panel)
        
        # Create class manager first
        self.class_manager = ClassManagerWidget()
        
        # Read initial data
        initial_run_id = self.run_ids[0]
        (initial_image, initial_masks) = self.read_data(initial_run_id)
        
        # Create the viewer with direct access to annotations dictionary
        self.segmentation_viewer = AnnotationSegmentationViewer(
            initial_image, 
            initial_masks,
            self.class_manager.get_class_dict(),
            self.class_manager.get_selected_class(),
            self.annotations,  # Pass the annotations dict directly
            initial_run_id
        )
        
        self.middle_layout.addWidget(self.segmentation_viewer)
        
        # Export/Import buttons
        bottom_layout = QHBoxLayout()
        
        self.export_json_btn = QPushButton("Save Annotations")
        self.import_json_btn = QPushButton("Load Annotations")
        
        bottom_layout.addWidget(self.import_json_btn)
        bottom_layout.addWidget(self.export_json_btn)
        
        self.middle_layout.addLayout(bottom_layout)
        self.main_splitter.addWidget(self.middle_panel)
        
        # --- Right Panel: Class Manager ---
        self.main_splitter.addWidget(self.class_manager)
        
        self.setCentralWidget(self.main_splitter)
        self.main_splitter.setSizes([125, 750, 200])
        self.resize(1100, 600)
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect all signals"""
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.export_json_btn.clicked.connect(self.export_annotations)
        self.import_json_btn.clicked.connect(self.import_annotations)
        
        self.class_manager.classAdded.connect(self.on_class_added)
        self.class_manager.classRemoved.connect(self.on_class_removed)
        self.class_manager.classSelected.connect(self.on_class_selected)
    
    def on_class_added(self, class_name, class_info):
        """Handle class addition"""
        self.segmentation_viewer.class_dict = self.class_manager.get_class_dict()
        print(f"Added class: {class_name}")
    
    def on_class_removed(self, class_name):
        """Handle class removal - remove all annotations for this class"""
        for run_id in self.annotations:
            masks_to_remove = [
                mask_idx for mask_idx, cls in self.annotations[run_id].items() 
                if cls == class_name
            ]
            for mask_idx in masks_to_remove:
                del self.annotations[run_id][mask_idx]
        
        self.segmentation_viewer.class_dict = self.class_manager.get_class_dict()
        print(f"Removed class: {class_name} and all its annotations")
    
    def on_class_selected(self, class_name):
        """Handle class selection"""
        self.segmentation_viewer.selected_class = class_name
        print(f"Selected class: {class_name}")
    
    def read_data(self, run_id):
        """Read the base image and segmentation masks for a given run ID"""
        base_image = self.root[run_id][0][:]
        try:
            masks = self.root[run_id]['labels'][0][:]
        except:
            masks = self.root[run_id]['masks'][:]
        
        (nx, ny) = base_image.shape
        if nx < ny:
            base_image = base_image.T
            masks = np.swapaxes(masks, 1, 2)
        
        return base_image, masks
    
    def on_image_selected(self, item):
        """Load the selected image into the viewer"""
        run_id = item.text()
        
        try:
            base_image, masks = self.read_data(run_id)
        except Exception as e:
            print(f"Error loading data for run ID {run_id}: {e}")
            return
        
        # Load data with the new run_id
        self.segmentation_viewer.load_data(
            base_image, 
            masks, 
            self.class_manager.get_class_dict(),
            run_id  # Pass the run_id
        )
        
        print(f"Loaded run {run_id}, annotations: {self.annotations[run_id]}")
    
    def export_annotations(self):
        """Export annotations to JSON file"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", 
            f"training.json",
            "JSON Files (*.json)"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            QMessageBox.information(self, "Success", f"Annotations saved to {filepath}")
            print(f"Exported annotations to: {filepath}")
    
    def import_annotations(self):
        """Import annotations from JSON file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations",
            "",
            "JSON Files (*.json)"
        )
        
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_annotations = json.load(f)
            
            # Update annotations
            self.annotations.update(loaded_annotations)
            
            # Extract all unique classes from the annotations
            all_classes = set()
            for run_annotations in self.annotations.values():
                for class_name in run_annotations.values():
                    all_classes.add(class_name)
            
            # Add any missing classes
            for class_name in all_classes:
                if class_name not in self.class_manager.class_dict:
                    color = self.class_manager.get_next_color(len(self.class_manager.class_dict))
                    self.class_manager.add_class_to_dict(class_name, color)
            
            # Reload current view
            current_item = self.image_list.currentItem()
            if current_item:
                self.on_image_selected(current_item)
            
            QMessageBox.information(self, "Success", f"Annotations loaded from {filepath}")
            print(f"Imported annotations from: {filepath}")
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        current_row = self.image_list.currentRow()
        
        if event.key() == Qt.Key_Left:
            new_row = current_row - 1
            self.load_next_runID(new_row)
        elif event.key() == Qt.Key_Right:
            new_row = current_row + 1
            self.load_next_runID(new_row)
        elif event.key() == Qt.Key_S:
            self.export_annotations()
        else:
            super().keyPressEvent(event)
    
    def load_next_runID(self, new_row):
        """Load the next run ID"""
        new_row = max(0, min(new_row, self.image_list.count() - 1))
        self.image_list.setCurrentRow(new_row)
        self.on_image_selected(self.image_list.item(new_row))
    
    def setup_menu_bar(self):
        """Sets up the menu bar with a 'Help' menu"""
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        welcome_action = help_menu.addAction("Show Welcome Message")
        welcome_action.triggered.connect(self.show_welcome_message)
    
    def show_welcome_message(self):
        """Displays a welcome message"""
        message = (
            "Welcome to the Dynamic SAM2-ET Annotation GUI!\n\n"
            "Quick Tutorial:\n"
            "1. **Managing Classes**:\n"
            "   - Add new classes using the panel on the right\n"
            "   - Each class gets a unique color automatically\n\n"
            "2. **Navigating Images**:\n"
            "   - Use Left/Right Arrow Keys to navigate\n\n"
            "3. **Annotating**:\n"
            "   - Select a class from the right panel\n"
            "   - Click on masks to assign them to the class\n"
            "   - Press 'R' to undo last assignment\n\n"
            "4. **Saving**:\n"
            "   - Press 'S' or click 'Export' to save as JSON\n"
            "   - Import previous annotations to continue work\n"
        )
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Welcome")
        msg_box.setText(message)
        msg_box.exec_()


@click.command(context_settings={"show_default": True})
@click.option('--input', type=str, required=True, 
              help="Path to the input Zarr file.")
def gui(input: str):
    """
    Dynamic GUI for annotating SAM2 segmentations with custom classes.
    """
    app = QApplication(sys.argv)
    main_window = MainWindow(input)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui()