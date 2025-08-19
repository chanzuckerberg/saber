import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from saber.gui.base.segmentation_picker import SegmentationViewer

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available, using slower boundary detection")

from typing import Dict, Optional, Sequence

# ---------- ViewBoxes ----------

class NoRightZoomViewBox(pg.ViewBox):
    """
    ViewBox that disables the default right-drag rubber-band zoom and context menu.
    Used for the RIGHT panel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMenuEnabled(False)

    def mouseDragEvent(self, ev, axis=None):
        # Swallow right-drag so rubber-band zoom never starts
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        # Swallow right-click entirely so nothing happens on press/release
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
            return
        super().mouseClickEvent(ev)


class LeftDrawViewBox(pg.ViewBox):
    """
    ViewBox for the LEFT panel:
      - Disables the default right-drag zoom
      - Reports right-press/drag/release back to the parent viewer to draw a circle
    """
    def __init__(self, viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMenuEnabled(False)
        self._viewer = viewer
        self._right_dragging = False

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self._right_dragging = True
            # use scene pos so the viewer can map to image coords reliably
            self._viewer._circle_drag_start(ev.scenePos())
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self._right_dragging and ev.button() == QtCore.Qt.RightButton:
            self._viewer._circle_drag_update(ev.scenePos())
            ev.accept()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseReleaseEvent(self, ev):
        if self._right_dragging and ev.button() == QtCore.Qt.RightButton:
            self._right_dragging = False
            self._viewer._circle_drag_finish(ev.scenePos())
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# ---------- Main widget ----------

class HashtagSegmentationViewer(pg.GraphicsLayoutWidget):
    """
    Enhanced SegmentationViewer with hashtag-based coloring, selection highlighting,
    and right-drag circle creation (LEFT panel) that appends a new segmentation mask.
    """

    # Minimum circle radius (in pixels) to create a mask on release
    MIN_CIRCLE_RADIUS_PX = 2.0

    maskAdded = QtCore.pyqtSignal(int)  # Signal to indicate a new mask has been added

    def __init__(self, image: np.ndarray, masks: Sequence[np.ndarray]):
        """
        image: 2D numpy array (Nx, Ny) - the background image
        masks: list/array of 2D masks (shape: (N_classes, Nx, Ny))
        """
        super().__init__()

        # ---- Data & state ----
        self.image = image
        # Internally keep masks as a list of 2D arrays; appending is frequent.
        self.masks = list(masks)

        self.accepted_masks = set()
        self.accepted_stack = []

        # Colors
        self.tab10_colors = SegmentationViewer.TAB10_COLORS
        self.custom_colors: Dict[int, tuple] = {}

        # Selection & boundary cache
        self.selected_mask_id: Optional[int] = None
        self.selection_boundary_item = None
        self.boundary_cache: Dict[int, Optional[np.ndarray]] = {}
        self._boundary_cache_keys = set()

        # Temporary circle overlay while dragging on the LEFT panel
        self._circle_center = None  # (x0, y0) in image coords (floats)
        self._temp_circle_item: Optional[QtGui.QGraphicsEllipseItem] = None

        # ---- Layout / Views ----
        # Left view: uses custom ViewBox that captures right-drag circle
        self.left_view = LeftDrawViewBox(self, lockAspect=True, enableMenu=False)
        # Right view: uses ViewBox that swallows right-clicks
        self.right_view = NoRightZoomViewBox(lockAspect=True, enableMenu=False)
        self.addItem(self.left_view, row=0, col=0)
        self.addItem(self.right_view, row=0, col=1)

        # Base images
        self.left_base_img_item = pg.ImageItem(self.image)
        self.right_base_img_item = pg.ImageItem(self.image)
        self.left_view.addItem(self.left_base_img_item)
        self.right_view.addItem(self.right_base_img_item)

        # Overlays for masks (parallel lists)
        self.left_mask_items = []
        self.right_mask_items = []

        # Click handling
        self.scene().sigMouseClicked.connect(self.mouse_clicked)

        # Focus for key handling
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()

        # NOTE: As in your current setup, overlays are expected to be created
        # by calling `initialize_overlays()` (e.g., via load_data_fresh()).
        # We do not auto-call it here to preserve your exact flow.

    # ---------- Public API ----------

    def update_mask_colors(self, color_mapping: Dict[int, str]):
        """Update mask overlay colors based on hashtag colors."""
        new_custom_colors = {}
        for mask_id, hex_color in color_mapping.items():
            hc = hex_color[1:] if hex_color.startswith('#') else hex_color
            try:
                r = int(hc[0:2], 16) / 255.0
                g = int(hc[2:4], 16) / 255.0
                b = int(hc[4:6], 16) / 255.0
                new_custom_colors[mask_id] = (r, g, b)
            except (ValueError, IndexError):
                print(f"Invalid hex color: {hex_color}")
                continue

        if new_custom_colors != self.custom_colors:
            self.custom_colors = new_custom_colors
            self.refresh_overlays()

    def load_data_fresh(self, base_image, masks):
        """Fresh data loading - clean slate approach."""
        # Clear caches/state
        self.boundary_cache.clear()
        self._boundary_cache_keys.clear()
        self.custom_colors.clear()
        self.clear_highlight()
        self.selected_mask_id = None

        # Data
        self.image = base_image
        self.masks = list(masks)

        # Reset accepted masks/stack
        self.accepted_masks.clear()
        self.accepted_stack.clear()

        # Update base images
        self.left_base_img_item.setImage(base_image)
        self.right_base_img_item.setImage(base_image)

        # Remove old overlays
        for item in getattr(self, 'left_mask_items', []):
            try:
                self.left_view.removeItem(item)
            except Exception:
                pass
        for item in getattr(self, 'right_mask_items', []):
            try:
                self.right_view.removeItem(item)
            except Exception:
                pass

        self.left_mask_items = []
        self.right_mask_items = []

        # Create overlays for all masks
        self.initialize_overlays()

    def initialize_overlays(self):
        """Create overlays for all masks (left shown, right hidden)."""
        for i, mask in enumerate(self.masks):
            # Reverse z-order: later masks (smaller area) get higher z-values (appear on top)
            z_value = len(self.masks) - i
            
            left_item = pg.ImageItem(self.create_overlay_rgba(mask, i))
            left_item.setOpacity(0.4)
            left_item.setZValue(z_value)  # Higher index = smaller area = higher z-value = on top
            left_item.setVisible(True)
            self.left_view.addItem(left_item)
            self.left_mask_items.append(left_item)

            right_item = pg.ImageItem(self.create_overlay_rgba(mask, i))
            right_item.setOpacity(0.4)
            right_item.setZValue(z_value)  # Same z-value for consistency
            right_item.setVisible(False)
            self.right_view.addItem(right_item)
            self.right_mask_items.append(right_item)

    def load_data(self, base_image, masks, class_dict=None):
        """Preserve parent behavior while clearing caches before load."""
        self.boundary_cache.clear()
        self._boundary_cache_keys.clear()
        self.custom_colors.clear()
        self.clear_highlight()
        # Delegate to parent (as in your code)
        super().load_data(base_image, masks, class_dict)

    # ---------- Rendering helpers ----------

    def create_overlay_rgba(self, mask, index=0):
        """
        Use custom colors when available; otherwise fall back to tab10.
        Assumes mask is shape (Nx, Ny) and indexes as [x, y].
        """
        Nx, Ny = mask.shape
        rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)

        color = self.custom_colors.get(index, self.tab10_colors[index % len(self.tab10_colors)])
        inds = mask > 0.5
        rgba[inds, 0] = color[0]
        rgba[inds, 1] = color[1]
        rgba[inds, 2] = color[2]
        rgba[inds, 3] = 1.0
        return rgba

    def refresh_overlays(self):
        """Refresh ALL mask overlays with current colors, regardless of visibility."""
        for i, mask in enumerate(self.masks):
            # Update left panel overlay (whether visible or not)
            if i < len(self.left_mask_items):
                self.left_mask_items[i].setImage(self.create_overlay_rgba(mask, i))
            
            # Update right panel overlay (whether visible or not)  
            if i < len(self.right_mask_items):
                self.right_mask_items[i].setImage(self.create_overlay_rgba(mask, i))
        
        print(f"Updated colors for {len(self.masks)} masks on both panels")

    # ---------- Selection / Highlight ----------

    def highlight_mask(self, mask_id: int):
        """Add a boundary highlight around a specific mask and update selected_mask_id."""
        if mask_id < 0 or mask_id >= len(self.masks):
            return
        self.clear_highlight()
        self.selected_mask_id = mask_id

        # Only highlight if the mask is visible on the right panel (i.e., accepted)
        if mask_id < len(self.right_mask_items) and self.right_mask_items[mask_id].isVisible():
            boundary_points = self.get_mask_boundary(mask_id)
            if boundary_points is None or len(boundary_points) == 0:
                return

            self.selection_boundary_item = pg.PlotDataItem(
                boundary_points[:, 1], boundary_points[:, 0],
                pen=pg.mkPen(color='white', width=3, style=QtCore.Qt.DashLine),
                connect='finite'
            )
            self.right_view.addItem(self.selection_boundary_item)

    def clear_highlight(self):
        """Clear the selection highlight and reset selected_mask_id."""
        if self.selection_boundary_item is not None:
            try:
                self.right_view.removeItem(self.selection_boundary_item)
            except Exception:
                try:
                    self.left_view.removeItem(self.selection_boundary_item)
                except Exception:
                    pass
            self.selection_boundary_item = None
        self.selected_mask_id = None

    def set_accepted_indices(self, indices):
        """Make accepted masks show on RIGHT only, others on LEFT only."""
        self.accepted_masks = set(map(int, indices))
        self.accepted_stack = list(self.accepted_masks)
        for i in range(len(self.masks)):
            on_right = i in self.accepted_masks
            if i < len(self.right_mask_items):
                self.right_mask_items[i].setVisible(on_right)
            if i < len(self.left_mask_items):
                self.left_mask_items[i].setVisible(not on_right)

    # ---------- Boundary extraction ----------

    def get_mask_boundary(self, mask_id: int) -> Optional[np.ndarray]:
        """Get boundary points for a mask using fast method with caching."""
        mask = self.masks[mask_id]
        cache_key = f"{mask_id}_{hash(str(mask.tobytes()))}"
        if cache_key in self._boundary_cache_keys:
            return self.boundary_cache.get(mask_id)

        if HAS_OPENCV:
            boundary_points = self._get_boundary_opencv_fast(mask)
        else:
            boundary_points = self._get_boundary_numpy_fast(mask)

        # Limit cache size
        if len(self.boundary_cache) > 50:
            # clear ~half
            keys_to_remove = list(self.boundary_cache.keys())[:25]
            for k in keys_to_remove:
                del self.boundary_cache[k]
            self._boundary_cache_keys = set(self.boundary_cache.keys())
        self.boundary_cache[mask_id] = boundary_points
        self._boundary_cache_keys.add(cache_key)
        return boundary_points

    def _get_boundary_opencv_fast(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Fast boundary detection using OpenCV with aggressive optimization."""
        try:
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
        except Exception as e:
            print(f"OpenCV boundary detection failed: {e}")
            return self._get_boundary_numpy_fast(mask)

    def _get_boundary_numpy_fast(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Fallback boundary detection without SciPy. Returns Nx2 array of [x, y] points."""
        m = mask > 0.5
        if not m.any():
            return None
        b = np.zeros_like(m, dtype=bool)
        # 4-neighborhood difference
        b[1:-1, 1:-1] = m[1:-1, 1:-1] & (
            (~m[0:-2, 1:-1]) | (~m[2:, 1:-1]) | (~m[1:-1, 0:-2]) | (~m[1:-1, 2:])
        )
        xs, ys = np.nonzero(b)
        if xs.size == 0:
            return None
        return np.column_stack([xs, ys])

    # ---------- Keyboard ----------

    def keyPressEvent(self, event):
        """
        Revert the currently selected/highlighted mask.
        Press 'r' => revert the currently selected/highlighted mask
        """
        key = event.key()
        if key == QtCore.Qt.Key_R:
            print(f"Selected mask ID: {self.selected_mask_id}")
            if self.selected_mask_id is not None:
                mask_id = self.selected_mask_id
                if (mask_id in self.accepted_masks and
                    mask_id < len(self.right_mask_items) and
                    self.right_mask_items[mask_id].isVisible()):
                    # Remove from accepted
                    self.accepted_masks.remove(mask_id)
                    if mask_id in self.accepted_stack:
                        self.accepted_stack.remove(mask_id)
                    # Show on left, hide on right
                    self.right_mask_items[mask_id].setVisible(False)
                    self.left_mask_items[mask_id].setVisible(True)
                    self.clear_highlight()
                    self.signal_segmentation_deselected()
                    print(f"Reverted selected mask {mask_id}: hidden on right, shown on left.")
                    print(f"Current accepted masks: {self.accepted_masks}")
                else:
                    print(f"Selected mask {mask_id} is not currently accepted, cannot revert.")

    # ---------- Click handling (unchanged left-click accept behavior) ----------

    def mouse_clicked(self, event):
        """Handle clicks on both panels (selection & accept)."""
        scene_pos = event.scenePos()
        left_image_pos = self.left_base_img_item.mapFromScene(scene_pos)
        right_image_pos = self.right_base_img_item.mapFromScene(scene_pos)

        Nx, Ny = self.image.shape[:2]
        left_x, left_y = int(left_image_pos.x()), int(left_image_pos.y())
        right_x, right_y = int(right_image_pos.x()), int(right_image_pos.y())

        clicked_on_right = False
        clicked_on_left = False

        # Right panel: select accepted mask under cursor
        if 0 <= right_x < Nx and 0 <= right_y < Ny:
            for i in range(len(self.masks)):
                if (i < len(self.right_mask_items) and
                    self.right_mask_items[i].isVisible() and
                    self.masks[i][right_x, right_y] > 0):
                    self.highlight_mask(i)
                    self.signal_segmentation_selected(i)
                    clicked_on_right = True
                    break

        # Left panel: accept on left-click
        if not clicked_on_right and 0 <= left_x < Nx and 0 <= left_y < Ny:
            clicked_on_left = True
            self.mouse_clicked_left_panel(event)

            # If mask just got accepted, highlight on right
            for i in range(len(self.masks)):
                if (self.masks[i][left_x, left_y] > 0 and
                    i < len(self.right_mask_items) and
                    self.right_mask_items[i].isVisible()):
                    self.highlight_mask(i)
                    self.signal_segmentation_selected(i)
                    break

        if not clicked_on_right and not clicked_on_left:
            self.clear_highlight()
            self.signal_segmentation_deselected()

    def mouse_clicked_left_panel(self, event):
        """Left panel: left-click 'accepts' the topmost *visible* mask under the cursor."""
        scene_pos = event.scenePos()
        image_pos = self.left_base_img_item.mapFromScene(scene_pos)
        x, y = int(image_pos.x()), int(image_pos.y())

        Nx, Ny = self.image.shape[:2]
        if not (0 <= x < Nx and 0 <= y < Ny):
            print(f"Clicked out of bounds: {x}, {y}")
            return

        # Only consider masks that are visible on the LEFT and cover the pixel.
        # Then prioritize by zValue (topmost first).
        candidates = [
            i for i in range(len(self.masks))
            if (i < len(self.left_mask_items)
                and self.left_mask_items[i].isVisible()
                and self.masks[i][x, y] > 0)
        ]
        if not candidates:
            print("No mask at clicked location.")
            return

        candidates.sort(key=lambda i: self.left_mask_items[i].zValue(), reverse=True)

        # Cycling logic:
        # Reset cycling if position changed OR the candidate set changed (e.g., new mask added).
        hits_sig = tuple(candidates)
        if getattr(self, '_last_click_pos', None) != (x, y) or getattr(self, '_last_hits_signature', None) != hits_sig:
            self._last_click_pos = (x, y)
            self._last_hits_signature = hits_sig
            self._current_mask_index = 0
        else:
            self._current_mask_index = (self._current_mask_index + 1) % len(candidates)

        i_hit = candidates[self._current_mask_index]

        # Left button => accept (hide left, show right)
        if event.button() == QtCore.Qt.LeftButton:
            if i_hit < len(self.left_mask_items) and self.left_mask_items[i_hit].isVisible():
                self.left_mask_items[i_hit].setVisible(False)
                self.right_mask_items[i_hit].setVisible(True)
                self.accepted_masks.add(i_hit)
                self.accepted_stack.append(i_hit)
                print(self.accepted_masks)
                print(f"Accepted mask {i_hit}: now hidden on left, shown on right.")
        print(f"Current accepted masks: {self.accepted_masks}")

    # ---------- Signals to outside ----------

    def signal_segmentation_selected(self, mask_id: int):
        """Signal that a segmentation was selected (for main window to handle)."""
        self.last_selected_mask_id = mask_id
        if hasattr(self, 'selection_callback') and self.selection_callback:
            self.selection_callback(mask_id)

    def signal_segmentation_deselected(self):
        """Signal that no segmentation is selected."""
        self.last_selected_mask_id = None
        if hasattr(self, 'deselection_callback') and self.deselection_callback:
            self.deselection_callback()

    def set_selection_callbacks(self, selection_callback=None, deselection_callback=None):
        """Set callback functions for when segmentations are selected/deselected."""
        self.selection_callback = selection_callback
        self.deselection_callback = deselection_callback
        

    # ---------- Right-drag circle (LEFT panel) ----------

    def _map_scene_to_image(self, scene_pos: QtCore.QPointF):
        """Map scene position to image coordinates (float), and check bounds."""
        img_pos = self.left_base_img_item.mapFromScene(scene_pos)
        x_f, y_f = float(img_pos.x()), float(img_pos.y())
        Nx, Ny = self.image.shape[:2]
        in_bounds = (0.0 <= x_f < float(Nx)) and (0.0 <= y_f < float(Ny))
        return x_f, y_f, in_bounds

    def _ensure_temp_circle_item(self):
        """Create the temporary ellipse item if needed (in LEFT view data coords)."""
        if self._temp_circle_item is None:
            self._temp_circle_item = QtWidgets.QGraphicsEllipseItem()
            self._temp_circle_item.setParentItem(self.left_view.childGroup)
            self._temp_circle_item.setZValue(1e6)  # draw on top

            # Visuals
            self._temp_circle_item.setPen(pg.mkPen('w', width=2, style=QtCore.Qt.DashLine))
            self._temp_circle_item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))

            # **Critical**: don't intercept any mouse events
            self._temp_circle_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self._temp_circle_item.setAcceptHoverEvents(False)
            self._temp_circle_item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            self._temp_circle_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)

            # Ensure pyqtgraph manages it but it never affects bounds
            try:
                self.left_view.addItem(self._temp_circle_item, ignoreBounds=True)
            except TypeError:
                self.left_view.addItem(self._temp_circle_item)

    def _set_temp_circle_geometry(self, x0: float, y0: float, r: float):
        """Update the temporary circle rect in data (image) coordinates."""
        rect = QtCore.QRectF(x0 - r, y0 - r, 2 * r, 2 * r)
        self._temp_circle_item.setRect(rect)

    def _remove_temp_circle(self):
        """Remove the temporary ellipse item, if present."""
        if self._temp_circle_item is not None:
            try:
                self.left_view.removeItem(self._temp_circle_item)
            except Exception:
                pass
            self._temp_circle_item = None

    def _circle_drag_start(self, scene_pos: QtCore.QPointF):
        """Right-press on LEFT panel: start circle preview if in-bounds."""
        x0, y0, ok = self._map_scene_to_image(scene_pos)
        if not ok:
            self._circle_center = None
            return
        self._circle_center = (x0, y0)
        self._ensure_temp_circle_item()
        # Draw a tiny circle initially (invisible small)
        self._set_temp_circle_geometry(x0, y0, 0.0)

    def _circle_drag_update(self, scene_pos: QtCore.QPointF):
        """Right-drag on LEFT panel: update circle preview."""
        if self._circle_center is None:
            return
        x0, y0 = self._circle_center
        x1, y1, _ = self._map_scene_to_image(scene_pos)
        dx, dy = (x1 - x0), (y1 - y0)
        r = float(np.hypot(dx, dy))
        self._ensure_temp_circle_item()
        self._set_temp_circle_geometry(x0, y0, r)

    def _circle_drag_finish(self, scene_pos: QtCore.QPointF):
        """Right-release on LEFT panel: finalize circle -> create & append mask."""
        if self._circle_center is None:
            self._remove_temp_circle()
            return

        x0, y0 = self._circle_center
        x1, y1, _ = self._map_scene_to_image(scene_pos)
        self._circle_center = None
        r = float(np.hypot(x1 - x0, y1 - y0))

        # Remove preview
        self._remove_temp_circle()

        if r < self.MIN_CIRCLE_RADIUS_PX:
            # Too small => ignore
            return

        # Create mask (shape (Nx, Ny); your code indexes masks as [x, y])
        Nx, Ny = self.image.shape[:2]
        xx = np.arange(Nx)[:, None]    # axis 0 == "x" in your code
        yy = np.arange(Ny)[None, :]    # axis 1 == "y"
        circle = ((xx - x0) ** 2 + (yy - y0) ** 2) <= (r ** 2)
        new_mask = circle.astype(np.float32)

        # Sanity: the center pixel should be inside the mask (if in-bounds)
        cx, cy = int(x0), int(y0)
        if 0 <= cx < Nx and 0 <= cy < Ny and new_mask[cx, cy] <= 0:
            # This should not happen with the construction above; print once if it does.
            print("Warning: new circle mask center is not inside the mask; "
                "check axis conventions.")

        new_index = self._append_new_mask(new_mask)
        print(f"Added circle mask #{new_index} at center=({x0:.1f}, {y0:.1f}), r={r:.1f}px. "
            f"Total masks: {len(self.masks)}")


    def _append_new_mask(self, mask: np.ndarray) -> int:
        """
        Append a new 2D mask to internal storage and create corresponding overlays.
        New mask is visible on LEFT and hidden on RIGHT, consistent with your behavior.
        """
        mask = np.asarray(mask, dtype=np.float32)
        Nx, Ny = self.image.shape[:2]
        if mask.shape != (Nx, Ny):
            raise ValueError(f"New mask shape {mask.shape} must match image shape {(Nx, Ny)}")

        # Append to model
        self.masks.append(mask)
        i = len(self.masks) - 1

        # LEFT overlay (visible) — put it on TOP
        left_item = pg.ImageItem(self.create_overlay_rgba(mask, i))
        left_item.setOpacity(0.4)
        left_item.setVisible(True)
        left_item.setZValue(self._next_top_z(left=True))   # <<< topmost
        self.left_view.addItem(left_item)
        self.left_mask_items.append(left_item)

        # RIGHT overlay (hidden) — zValue doesn't matter but keep consistent
        right_item = pg.ImageItem(self.create_overlay_rgba(mask, i))
        right_item.setOpacity(0.4)
        right_item.setVisible(False)
        right_item.setZValue(self._next_top_z(left=False))
        self.right_view.addItem(right_item)
        self.right_mask_items.append(right_item)

        # Ensure caches know about the new entry (no boundary yet)
        self.boundary_cache.pop(i, None)

        self.maskAdded.emit(i)  # Signal to indicate a new mask has been added
        
        return i
    
    def _next_top_z(self, left: bool = True) -> float:
        """Return a z-value just above the current top item for the given panel."""
        items = self.left_mask_items if left else self.right_mask_items
        base = self.left_base_img_item if left else self.right_base_img_item
        if items:
            z_top = max((it.zValue() for it in items if it is not None), default=base.zValue())
        else:
            z_top = base.zValue()
        return z_top + 1.0