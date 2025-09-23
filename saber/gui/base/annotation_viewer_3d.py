from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import cv2
import time
from typing import Optional, Dict, Any

class AnnotationSegmentationViewer3D(QtWidgets.QWidget):
    """
    Fast 3D segmentation viewer using a label image + color LUT.
    - Only 2 mask ImageItems total (left/right)
    - O(1) picking via label map
    - Debounced slider updates
    - Highlight and remove functionality for annotated masks
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

        # Track highlighted label
        self.highlighted_label = None
        self.boundary_items = {'left': None, 'right': None}

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

        # 🔧 Align Qt coordinates with numpy indexing (top-left origin)
        self.left_view.invertY(True)
        self.right_view.invertY(True)

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

        # Create boundary items for highlighting
        self.boundary_items['left'] = pg.PlotDataItem(
            pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.SolidLine)
        )
        self.boundary_items['left'].setZValue(1000)
        self.boundary_items['left'].setVisible(False)
        self.left_view.addItem(self.boundary_items['left'])

        self.boundary_items['right'] = pg.PlotDataItem(
            pen=pg.mkPen(color='w', width=2, style=QtCore.Qt.SolidLine)
        )
        self.boundary_items['right'].setZValue(1000)
        self.boundary_items['right'].setVisible(False)
        self.right_view.addItem(self.boundary_items['right'])

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
        # Update boundary if a label is highlighted
        if self.highlighted_label is not None:
            self._update_highlight_boundary()

    def _display_current_slice(self):
        # Update base images
        current_image = self.volume_3d[self.current_slice]
        self.left_base_img.setImage(current_image, autoLevels=False, axisOrder='row-major')
        self.right_base_img.setImage(current_image, autoLevels=False, axisOrder='row-major')

        # Build per-panel LUTs from current annotations
        left_lut, right_lut = self._make_left_right_luts()

        # Show the same label slice with different LUTs
        label_slice = self.labels_3d[self.current_slice]
        levels = (0, max(1, self.max_label))
        self.left_mask_img.setImage(label_slice, levels=levels, lut=left_lut,
                                    autoLevels=False, axisOrder='row-major')
        self.right_mask_img.setImage(label_slice, levels=levels, lut=right_lut,
                                    autoLevels=False, axisOrder='row-major')

    def highlight_label(self, label_value):
        """Highlight a specific label with boundary on the appropriate panel"""
        self.clear_highlight()
        self.highlighted_label = label_value
        self._update_highlight_boundary()

    def _update_highlight_boundary(self):
        """Update boundary display for the currently highlighted label"""
        if self.highlighted_label is None:
            self.clear_highlight()
            return

        # Get current slice labels
        label_slice = self.labels_3d[self.current_slice]
        
        # Create binary mask for this label
        mask = (label_slice == self.highlighted_label).astype(np.float32)
        
        if not mask.any():
            # Label not present in current slice
            self.boundary_items['left'].setVisible(False)
            self.boundary_items['right'].setVisible(False)
            return

        # Get boundary points
        boundary_pts = self._get_boundary_opencv_fast(mask)
        
        if boundary_pts is None or len(boundary_pts) == 0:
            self.boundary_items['left'].setVisible(False)
            self.boundary_items['right'].setVisible(False)
            return

        # Close the contour
        boundary_pts = np.vstack([boundary_pts, boundary_pts[0:1]])

        # Align contour (pixel centers) to ImageItem's unit-square geometry (pixel corners)
        # OpenCV returns (y, x) after our squeeze; we plot as (x, y).
        xs = boundary_pts[:, 0].astype(float) + 0.5
        ys = boundary_pts[:, 1].astype(float) + 0.5
        
        # Determine which panel to show boundary on
        run_annotations = self.annotations_dict.get(self.current_run_id, {})
        label_str = str(self.highlighted_label)
        
        if label_str in run_annotations:
            # Annotated - show on right panel
            self.boundary_items['right'].setData(xs, ys)
            self.boundary_items['right'].setVisible(True)
            self.boundary_items['left'].setVisible(False)
        else:
            # Not annotated - show on left panel
            self.boundary_items['left'].setData(xs, ys)
            self.boundary_items['left'].setVisible(True)
            self.boundary_items['right'].setVisible(False)

    def clear_highlight(self):
        """Clear all boundary highlights"""
        self.boundary_items['left'].setVisible(False)
        self.boundary_items['right'].setVisible(False)
        self.highlighted_label = None

    def mouse_clicked(self, event):
        """Handle clicks on both panels - left for annotation, right for selection"""
        if event.button() != QtCore.Qt.LeftButton:
            return

        scene_pos = event.scenePos()
        
        # Check if click is in left view
        if self.left_view.sceneBoundingRect().contains(scene_pos):
            img_pos = self.left_mask_img.mapFromScene(scene_pos)
            col_f, row_f = img_pos.x(), img_pos.y()
            label_slice = self.labels_3d[self.current_slice]
            rows, cols = label_slice.shape

            col = int(col_f)    # floor, more stable near edges
            row = int(row_f)
            if col < 0 or row < 0 or col >= cols or row >= rows:
                return

            val = int(label_slice[row, col])
            if val <= 0:
                return

            run_map = self.annotations_dict.setdefault(self.current_run_id, {})
            run_map[str(val)] = self.selected_class
            if val not in self.class_dict[self.selected_class]['masks']:
                self.class_dict[self.selected_class]['masks'].append(val)

            self.highlight_label(val)
            self._display_current_slice()
            self._update_highlight_boundary()

        elif self.right_view.sceneBoundingRect().contains(scene_pos):
            img_pos = self.right_mask_img.mapFromScene(scene_pos)
            col_f, row_f = img_pos.x(), img_pos.y()
            label_slice = self.labels_3d[self.current_slice]
            rows, cols = label_slice.shape

            col = int(col_f)
            row = int(row_f)
            if col < 0 or row < 0 or col >= cols or row >= rows:
                return

            val = int(label_slice[row, col])
            if val <= 0:
                return

            run_annotations = self.annotations_dict.get(self.current_run_id, {})
            label_str = str(val)
            if label_str in run_annotations:
                if self.highlighted_label == val:
                    self.clear_highlight()
                else:
                    self.highlight_label(val)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_R:
            # Remove the currently highlighted mask
            if self.highlighted_label is None:
                print("No mask selected to remove")
                return

            label_value = self.highlighted_label
            label_str = str(label_value)

            # Check if this label is actually annotated
            run_annotations = self.annotations_dict.get(self.current_run_id, {})
            if label_str not in run_annotations:
                print(f"Label {label_value} is not annotated")
                return

            # Get the class name before removing
            class_name = run_annotations[label_str]
            
            # Remove from annotations dict
            del run_annotations[label_str]
            
            # Remove from class dict
            if class_name in self.class_dict:
                if label_value in self.class_dict[class_name]['masks']:
                    self.class_dict[class_name]['masks'].remove(label_value)

            # Clear highlight
            self.clear_highlight()

            # Refresh display
            self._display_current_slice()
            
            print(f"Removed label {label_value} (class: {class_name})")

        elif event.key() == QtCore.Qt.Key_Up:
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

        # Clear highlight
        self.clear_highlight()

        # Clear reverse mapping (class -> masks), will be repopulated by load_existing_annotations
        for class_name in self.class_dict:
            self.class_dict[class_name]['masks'].clear()

        # Update view
        self._display_current_slice()
        self.load_existing_annotations()

        # Keep the existing view range/aspect; don't autoRange every time