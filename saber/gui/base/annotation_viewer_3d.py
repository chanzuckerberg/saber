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
import time

class AnnotationSegmentationViewer3D(QtWidgets.QWidget):
    """
    Fast 3D segmentation viewer using a label image + color LUT.
    - Only 2 mask ImageItems total (left/right)
    - O(1) picking via label map
    - Debounced slider updates
    """

    def __init__(self, volume, masks, class_dict, selected_class, annotations_dict, current_run_id):
        super().__init__()

        t0 = time.time()
        self.class_dict = class_dict
        self.selected_class = selected_class
        self.annotations_dict = annotations_dict
        self.current_run_id = current_run_id

        # 3D data (grayscale volume)
        self.volume_3d = volume
        self.n_slices = volume.shape[0]
        self.current_slice = self.n_slices // 2

        # Build (or accept) label map once
        self.labels_3d, self.max_label = self._prepare_label_volume(masks)

        # Precompute default palette LUT (unannotated colors)
        self.default_palette_lut = self._build_default_palette_lut(self.max_label)

        # UI & items
        self._setup_ui()

        # Initial display
        self._display_current_slice()

        # Load any existing annotations for current run
        self.load_existing_annotations()

        print(f"3D viewer ready in {time.time() - t0:.2f}s")

    # ---------- data prep ----------
    def _prepare_label_volume(self, masks):
        """
        Produce a single int label volume from input:
        - If masks is label map: use directly (as int32)
        - If masks is stack of binary masks: collapse to label map (1-based)
        """
        masks = np.asarray(masks)
        if masks.ndim == 3 and np.issubdtype(masks.dtype, np.integer):
            # Already (nz, nx, ny) labels
            labels = masks.astype(np.int32, copy=False)
            max_label = int(labels.max()) if labels.size else 0
            return labels, max_label

        if masks.ndim == 4:
            # (n_masks, nz, nx, ny) -> label map
            n_masks, nz, nx, ny = masks.shape
            labels = np.zeros((nz, nx, ny), dtype=np.int32)
            # Assign last-one-wins for overlapping; typical SAM2-like masks are disjoint
            for i in range(n_masks):
                # mask is {0,1} (or >0), convert to boolean
                m = masks[i] > 0
                labels[m] = i + 1  # 1-based
            return labels, n_masks

        raise ValueError(f"Unsupported masks shape: {masks.shape}")

    def _build_default_palette_lut(self, max_label):
        """A static LUT for unannotated masks using a tab10-ish repeating palette."""
        TAB10 = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [0, 128, 128],
            [188, 189, 34],
            [23, 190, 207],
        ], dtype=np.uint8)

        lut = np.zeros((max_label + 1, 4), dtype=np.uint8)
        lut[0] = [0, 0, 0, 0]  # background transparent
        if max_label > 0:
            reps = (max_label + 9) // 10
            palette = np.vstack([TAB10] * reps)[:max_label]
            lut[1:, :3] = palette
            lut[1:, 3] = 128  # alpha
        return lut

    def _make_left_right_luts(self):
        """
        Build two LUTs:
        - left_lut: default colors for unannotated, transparent for annotated labels
        - right_lut: transparent for unannotated, class color for annotated labels
        """
        left_lut = self.default_palette_lut.copy()
        right_lut = np.zeros_like(left_lut, dtype=np.uint8)  # default transparent
        # Always keep background transparent
        left_lut[0] = [0, 0, 0, 0]
        right_lut[0] = [0, 0, 0, 0]

        run_annotations = self.annotations_dict.get(self.current_run_id, {})
        if not run_annotations:
            return left_lut, right_lut

        for label_str, class_name in run_annotations.items():
            try:
                label = int(label_str)
            except Exception:
                continue
            if 0 < label <= self.max_label:
                # Left: annotated -> transparent
                left_lut[label] = [0, 0, 0, 0]
                # Right: annotated -> class color
                if class_name in self.class_dict:
                    c = self.class_dict[class_name]['color']
                    right_lut[label] = [c.red(), c.green(), c.blue(), 128]
        return left_lut, right_lut

    # ---------- UI ----------
    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.graphics_widget = pg.GraphicsLayoutWidget()

        self.left_view = self.graphics_widget.addViewBox(row=0, col=0)
        self.right_view = self.graphics_widget.addViewBox(row=0, col=1)

        self.left_view.setAspectLocked(True)
        self.right_view.setAspectLocked(True)
        self.left_view.setXLink(self.right_view)
        self.left_view.setYLink(self.right_view)

        # Base grayscale images
        initial = self.volume_3d[self.current_slice]
        self.left_base_img = pg.ImageItem(initial)
        self.right_base_img = pg.ImageItem(initial)
        self.left_view.addItem(self.left_base_img)
        self.right_view.addItem(self.right_base_img)

        # Mask overlays (only ONE per view)
        self.left_mask_img = pg.ImageItem()
        self.right_mask_img = pg.ImageItem()
        self.left_mask_img.setZValue(1)
        self.right_mask_img.setZValue(1)
        self.left_mask_img.setOpacity(1.0)
        self.right_mask_img.setOpacity(1.0)
        self.left_view.addItem(self.left_mask_img)
        self.right_view.addItem(self.right_mask_img)

        # auto-range once
        self.left_view.autoRange()

        # clicking: connect on the scene
        self.graphics_widget.scene().sigMouseClicked.connect(self.mouse_clicked)

        main_layout.addWidget(self.graphics_widget, stretch=1)

        # Slider (debounced)
        self._create_slider()
        main_layout.addWidget(self.slider_widget)

    def _create_slider(self):
        self.slider_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 10, 5, 10)

        self.slice_label = QtWidgets.QLabel(f'{self.current_slice + 1}')
        self.slice_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slice_label.setStyleSheet("font-weight: bold; font-size: 11px;")

        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.n_slices - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.setInvertedAppearance(True)
        # Smooth dragging without spamming updates:
        self.slice_slider.setTracking(False)  # update on release
        self.slice_slider.valueChanged.connect(self.on_slice_changed)

        layout.addWidget(self.slice_label)
        layout.addWidget(self.slice_slider, stretch=1)

        self.slider_widget.setLayout(layout)
        self.slider_widget.setMaximumWidth(80)

    # ---------- display & interaction ----------
    def on_slice_changed(self, value):
        self.current_slice = value
        self.slice_label.setText(f'{value + 1}')
        self._display_current_slice()

    def _display_current_slice(self):
        # Update base images
        current_image = self.volume_3d[self.current_slice]
        self.left_base_img.setImage(current_image, autoLevels=False)
        self.right_base_img.setImage(current_image, autoLevels=False)

        # Build per-panel LUTs from current annotations
        left_lut, right_lut = self._make_left_right_luts()

        # Show the same label slice with different LUTs
        label_slice = self.labels_3d[self.current_slice]
        levels = (0, max(1, self.max_label))
        self.left_mask_img.setImage(label_slice, levels=levels, lut=left_lut, autoLevels=False)
        self.right_mask_img.setImage(label_slice, levels=levels, lut=right_lut, autoLevels=False)

    def mouse_clicked(self, event):
        """Assign clicked label to the selected class (left view only)."""
        if not self.selected_class or self.selected_class not in self.class_dict:
            return

        scene_pos = event.scenePos()
        if not self.left_view.sceneBoundingRect().contains(scene_pos):
            return

        img_pos = self.left_base_img.mapFromScene(scene_pos)
        x, y = int(img_pos.x()), int(img_pos.y())  # note: x=col, y=row

        # Bounds check: label_slice is (rows, cols)
        label_slice = self.labels_3d[self.current_slice]
        h, w = label_slice.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        val = int(label_slice[y, x])
        if val <= 0:
            return

        # Record annotation
        run_map = self.annotations_dict.setdefault(self.current_run_id, {})
        run_map[str(val)] = self.selected_class

        # Maintain reverse list inside class_dict (optional bookkeeping)
        if val not in self.class_dict[self.selected_class]['masks']:
            self.class_dict[self.selected_class]['masks'].append(val)

        # Refresh LUTs/view
        self._display_current_slice()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up:
            if self.current_slice < self.n_slices - 1:
                self.slice_slider.setValue(self.current_slice + 1)
        elif event.key() == QtCore.Qt.Key_Down:
            if self.current_slice > 0:
                self.slice_slider.setValue(self.current_slice - 1)
        else:
            super().keyPressEvent(event)

    def load_existing_annotations(self):
        """Restore class -> mask bookkeeping from annotations_dict (optional)."""
        if self.current_run_id not in self.annotations_dict:
            return

        # Clear class masks
        for class_name in self.class_dict:
            self.class_dict[class_name]['masks'].clear()

        # Rebuild reverse mapping
        for mask_value_str, class_name in self.annotations_dict[self.current_run_id].items():
            try:
                mask_value = int(mask_value_str)
            except Exception:
                continue
            if class_name in self.class_dict:
                self.class_dict[class_name]['masks'].append(mask_value)

        self._display_current_slice()

    def load_data(self, base_image, masks, class_dict, run_id):
        """Load a new volume + masks."""
        self.current_run_id = run_id
        self.volume_3d = base_image
        self.n_slices = base_image.shape[0]
        self.class_dict = class_dict

        self.labels_3d, self.max_label = self._prepare_label_volume(masks)

        # Reset slice position
        self.current_slice = self.n_slices // 2
        # update slider range/value without flooding signals
        old_block = self.slice_slider.blockSignals(True)
        self.slice_slider.setMaximum(self.n_slices - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.blockSignals(old_block)

        # Clear reverse mapping (class -> masks), will be repopulated by load_existing_annotations
        for class_name in self.class_dict:
            self.class_dict[class_name]['masks'].clear()

        # Update view
        self._display_current_slice()
        self.load_existing_annotations()

        # Keep the existing view range/aspect; don't autoRange every time

