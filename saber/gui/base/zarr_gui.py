from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QListWidget, QPlainTextEdit,
    QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QComboBox, QMessageBox,
    QLineEdit, QListWidgetItem, QInputDialog, QColorDialog, QFileDialog
)
from saber.gui.base.annotation_viewer_3d import AnnotationSegmentationViewer3D
from saber.gui.base.annotation_viewer import AnnotationSegmentationViewer
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from concurrent.futures import ThreadPoolExecutor
import sys, zarr, click, json, os
import numpy as np


class ClassManagerWidget(QWidget):
    """Widget for managing segmentation classes dynamically"""

    classAdded = pyqtSignal(str, dict)
    classRemoved = pyqtSignal(str)
    classSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.class_dict = {}
        self.selected_class = None
        self.used_color_indices = set()
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
        pass

    def get_next_available_color_index(self):
        idx = 0
        while idx in self.used_color_indices:
            idx += 1
        return idx

    def get_next_color(self, index):
        TAB10_COLORS = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (0, 128, 128),
            (188, 189, 34), (23, 190, 207),
        ]
        r, g, b = TAB10_COLORS[index % len(TAB10_COLORS)]
        return QColor(r, g, b)

    def add_class(self):
        name = self.class_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a class name.")
            return
        if name in self.class_dict:
            QMessageBox.warning(self, "Warning", f"Class '{name}' already exists.")
            return
        color_index = self.get_next_available_color_index()
        color = self.get_next_color(color_index)
        self.add_class_to_dict(name, color, color_index)
        self.class_input.clear()

    def add_class_to_dict(self, class_name, color, color_index=None):
        if color_index is None:
            color_index = self.get_next_available_color_index()
        self.used_color_indices.add(color_index)
        self.class_dict[class_name] = {
            'value': color_index + 1,
            'color': color,
            'color_index': color_index,
            'masks': []
        }
        item = QListWidgetItem(class_name)
        pix = QPixmap(16, 16); pix.fill(color)
        item.setIcon(QIcon(pix))
        self.class_list.addItem(item)
        self.class_list.setCurrentItem(item)
        self.on_class_selected(item)
        self.classAdded.emit(class_name, self.class_dict[class_name])

    def remove_class(self):
        current_item = self.class_list.currentItem()
        if not current_item:
            return
        class_name = current_item.text()
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Remove class '{class_name}' and all assignments?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            color_index = self.class_dict[class_name]['color_index']
            self.used_color_indices.discard(color_index)
            del self.class_dict[class_name]
            self.class_list.takeItem(self.class_list.row(current_item))
            self.classRemoved.emit(class_name)
            if self.class_list.count() > 0:
                self.class_list.setCurrentRow(0)
                self.on_class_selected(self.class_list.currentItem())
            else:
                self.selected_class = None
                self.remove_btn.setEnabled(False)

    def on_class_selected(self, item):
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

        # Add Rotation Flag
        self.apply_rotation = False

        # Load Zarr data
        if os.path.exists(zarr_path):
            self.root = zarr.open(zarr_path, mode='r')
        else:
            raise FileNotFoundError(f"Zarr file {zarr_path} does not exist.")
        self.run_ids = list(self.root.keys())
        if not self.run_ids:
            raise RuntimeError("No groups found in Zarr file.")

        # Init annotations: one dict per run_id
        self.annotations = {run_id: {} for run_id in self.run_ids}

        self.setWindowTitle("Saber Annotation GUI")
        self.main_splitter = QSplitter(Qt.Horizontal, self)

        # Left panel
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.image_list = QListWidget()
        for image_name in self.run_ids:
            self.image_list.addItem(image_name)
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
        self.left_layout.addWidget(self.image_list)
        self.main_splitter.addWidget(self.left_panel)

        # Middle panel
        self.middle_panel = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_panel)

        # Class manager first
        self.class_manager = ClassManagerWidget()

        # Read initial data (sync for the first one)
        initial_run_id = self.run_ids[0]
        (initial_image, initial_masks) = self.read_data(initial_run_id)

        # 2D/RGB vs 3D switch
        if initial_image.ndim == 2 or (initial_image.ndim == 3 and initial_image.shape[0] == 3):
            print('2D Images are Loaded')
            self.segmentation_viewer = AnnotationSegmentationViewer(
                initial_image,
                initial_masks,
                self.class_manager.get_class_dict(),
                self.class_manager.get_selected_class(),
                self.annotations,
                initial_run_id
            )
        else:
            print('3D Volumes are Loaded')
            self.segmentation_viewer = AnnotationSegmentationViewer3D(
                initial_image,
                initial_masks,
                self.class_manager.get_class_dict(),
                self.class_manager.get_selected_class(),
                self.annotations,
                initial_run_id
            )
            self.middle_layout.addWidget(self.segmentation_viewer)

        self.middle_layout.addWidget(self.segmentation_viewer)

        # # Import/Export
        # bottom_layout = QHBoxLayout()
        # self.export_json_btn = QPushButton("Export Annotations to JSON")
        # self.import_json_btn = QPushButton("Import Annotations from JSON")
        # bottom_layout.addWidget(self.import_json_btn)
        # bottom_layout.addWidget(self.export_json_btn)
        # self.middle_layout.addLayout(bottom_layout)

        # Control buttons
        bottom_layout = QHBoxLayout()
        
        # Rotation button
        self.rotate_btn = QPushButton("Rotate")
        self.rotate_btn.setCheckable(True)
        self.rotate_btn.clicked.connect(self.toggle_rotation)
        bottom_layout.addWidget(self.rotate_btn)
        
        # Add spacer to separate rotation from import/export
        # bottom_layout.addStretch()
        
        self.import_json_btn = QPushButton("Import Annotations from JSON")
        self.export_json_btn = QPushButton("Export Annotations to JSON")
        bottom_layout.addWidget(self.import_json_btn)
        bottom_layout.addWidget(self.export_json_btn)
        self.middle_layout.addLayout(bottom_layout)
        self.main_splitter.addWidget(self.middle_panel)

        # Right panel (class manager)
        self.main_splitter.addWidget(self.class_manager)

        self.setCentralWidget(self.main_splitter)
        self.main_splitter.setSizes([125, 750, 200])
        self.resize(1100, 600)

        # Connect signals
        self.connect_signals()

        # Prefetcher
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._cache = {}  # run_id -> Future[(base_image, masks)]
        if self.run_ids:
            self.prefetch(self.run_ids[0])
            if len(self.run_ids) > 1:
                self.prefetch(self.run_ids[1])

    def toggle_rotation(self):
        """Toggle rotation state and reload all images"""
        self.apply_rotation = not self.apply_rotation
        
        # Update button text
        if self.apply_rotation:
            self.rotate_btn.setText("Rotate: ON")
        else:
            self.rotate_btn.setText("Rotate: OFF")
        
        # Clear the cache to force reload with new rotation state
        self._cache.clear()
        
        # Reload current image
        current_item = self.image_list.currentItem()
        if current_item:
            run_id = current_item.text()
            base_image, masks = self.get_data(run_id)
            self.segmentation_viewer.load_data(
                base_image,
                masks,
                self.class_manager.get_class_dict(),
                run_id
            )
        
        # Prefetch next images with new rotation state
        if self.run_ids:
            idx = self.run_ids.index(run_id) if current_item else 0
            if idx + 1 < len(self.run_ids):
                self.prefetch(self.run_ids[idx + 1])
        
        print(f"Rotation {'enabled' if self.apply_rotation else 'disabled'} for all images")          

    def connect_signals(self):
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.export_json_btn.clicked.connect(self.export_annotations)
        self.import_json_btn.clicked.connect(self.import_annotations)
        self.class_manager.classAdded.connect(self.on_class_added)
        self.class_manager.classRemoved.connect(self.on_class_removed)
        self.class_manager.classSelected.connect(self.on_class_selected)

    # -------- Prefetch helpers --------

    def _read_pair(self, run_id):
        # Use the same orientation logic as startup to keep every slice consistent.
        return self.read_data(run_id)

    def prefetch(self, run_id):
        if run_id in self._cache:
            return
        self._cache[run_id] = self._pool.submit(self._read_pair, run_id)

    def get_data(self, run_id):
        fut = self._cache.get(run_id)
        if fut is None:
            return self._read_pair(run_id)  # sync fallback
        data = fut.result()
        # queue next
        idx = self.run_ids.index(run_id)
        if idx + 1 < len(self.run_ids):
            self.prefetch(self.run_ids[idx + 1])
        return data

    # -------- Class manager hooks --------

    def on_class_added(self, class_name, class_info):
        self.segmentation_viewer.class_dict = self.class_manager.get_class_dict()

    def on_class_removed(self, class_name):
        # Remove all annotations for this class across all runs
        for run_id in self.annotations:
            to_remove = [k for k, v in self.annotations[run_id].items() if v == class_name]
            for k in to_remove:
                del self.annotations[run_id][k]

        # Update viewer's class dict with new color assignments
        self.segmentation_viewer.class_dict = self.class_manager.get_class_dict()

        # Reload current view (use cache path)
        current_item = self.image_list.currentItem()
        if current_item:
            run_id = current_item.text()
            try:
                base_image, masks = self.get_data(run_id)
                self.segmentation_viewer.load_data(
                    base_image,
                    masks,
                    self.class_manager.get_class_dict(),
                    run_id
                )
            except Exception as e:
                print(f"Error reloading after class removal: {e}")

    def on_class_selected(self, class_name):
        self.segmentation_viewer.selected_class = class_name

    # -------- Data I/O (sync fallback used at startup only) --------

    def read_data(self, run_id):
        base_image = self.root[run_id][0][:]
        try:
            masks = self.root[run_id]['labels'][0][:]
        except Exception:
            masks = self.root[run_id]['masks'][:]

        if self.apply_rotation:
            base_image, masks = self._apply_rotations(base_image, masks)

        return base_image, masks

    def _apply_rotations(self, image, masks):
        if image.ndim == 2:
            # 2D grayscale
            image = np.rot90(image, k=-1)
            if masks.ndim == 2:
                masks = np.rot90(masks, k=-1)
            else:
                masks = np.rot90(masks, k=-1, axes=(1, 2))
        elif image.ndim == 3 and image.shape[0] == 3:
            # 2D RGB
            image = np.rot90(image, k=-1, axes=(1, 2))
            if masks.ndim == 2:
                masks = np.rot90(masks, k=-1)
            else:
                masks = np.rot90(masks, k=-1, axes=(1, 2))
        elif image.ndim == 3:
            # 3D volume
            image = np.rot90(image, k=-1, axes=(1, 2))
            if isinstance(masks, list):
                masks = [np.rot90(m, k=-1, axes=(1, 2)) if m.ndim == 3 
                        else np.rot90(m, k=-1) for m in masks]
            else:
                masks = np.rot90(masks, k=-1, axes=(-2, -1))

        return image, masks

    # -------- UI events --------

    def on_image_selected(self, item):
        run_id = item.text()
        try:
            base_image, masks = self.get_data(run_id)
        except Exception as e:
            print(f"Error loading data for run ID {run_id}: {e}")
            return

        import time
        t0 = time.time()
        self.segmentation_viewer.load_data(
            base_image,
            masks,
            self.class_manager.get_class_dict(),
            run_id
        )
        t1 = time.time()
        print(f'It took {t1-t0:.4f} seconds to load!')

        # prefetch next
        idx = self.run_ids.index(run_id)
        if idx + 1 < len(self.run_ids):
            self.prefetch(self.run_ids[idx + 1])

    def export_annotations(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", "labels.json", "JSON Files (*.json)"
        )
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            QMessageBox.information(self, "Success", f"Annotations saved to {filepath}")

    def import_annotations(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", "", "JSON Files (*.json)"
        )
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_annotations = json.load(f)
            self.annotations.update(loaded_annotations)

            # Ensure all classes exist
            all_classes = set()
            for run_annotations in self.annotations.values():
                for class_name in run_annotations.values():
                    all_classes.add(class_name)
            for class_name in all_classes:
                if class_name not in self.class_manager.class_dict:
                    color = self.class_manager.get_next_color(len(self.class_manager.class_dict))
                    self.class_manager.add_class_to_dict(class_name, color)

            # Reload current
            current_item = self.image_list.currentItem()
            if current_item:
                self.on_image_selected(current_item)

            QMessageBox.information(self, "Success", f"Annotations loaded from {filepath}")

    def keyPressEvent(self, event):
        current_row = self.image_list.currentRow()
        if event.key() in (Qt.Key_Left, Qt.Key_A):
            self.load_next_runID(current_row - 1)
        elif event.key() in (Qt.Key_Right, Qt.Key_D):
            self.load_next_runID(current_row + 1)
        elif event.key() in (Qt.Key_Up, Qt.Key_W):
            r = self.class_manager.class_list.currentRow()
            if r > 0:
                self.class_manager.class_list.setCurrentRow(r - 1)
                it = self.class_manager.class_list.currentItem()
                if it:
                    self.class_manager.on_class_selected(it)
        elif event.key() in (Qt.Key_Down, Qt.Key_S):
            r = self.class_manager.class_list.currentRow()
            if r < self.class_manager.class_list.count() - 1:
                self.class_manager.class_list.setCurrentRow(r + 1)
                it = self.class_manager.class_list.currentItem()
                if it:
                    self.class_manager.on_class_selected(it)
        else:
            super().keyPressEvent(event)

    def load_next_runID(self, new_row):
        new_row = max(0, min(new_row, self.image_list.count() - 1))
        self.image_list.setCurrentRow(new_row)
        self.on_image_selected(self.image_list.item(new_row))

    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        welcome_action = help_menu.addAction("Show Welcome Message")
        welcome_action.triggered.connect(self.show_welcome_message)

    def show_welcome_message(self):
        message = (
            "Welcome to the Saber Annotation GUI!\n\n"
            "Quick Tutorial:\n"
            "1. Managing Classes:\n"
            "   - Add new classes using the panel on the right\n"
            "   - Each class gets a unique color automatically\n\n"
            "2. Navigating Images:\n"
            "   - Use Left/Right or (A/D) Arrow Keys to navigate\n\n"
            "3. Annotating:\n"
            "   - Select a class from the right panel\n"
            "   - Click on masks to assign them to the class\n"
            "   - Press 'R' to undo last assignment\n\n"
            "   - Press 'W' or 'S' keys to navigate the class list\n\n"
            "4. Saving:\n"
            "   - Click 'Export' to save as JSON\n"
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
    Saber GUI for annotating SAM2 segmentations with custom classes.
    """
    app = QApplication(sys.argv)
    main_window = MainWindow(input)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui()