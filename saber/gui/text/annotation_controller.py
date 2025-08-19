"""
Controller for SAM2-ET text annotation UI with run-level annotations.
Each run loads and saves its own annotations independently.
"""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox
from typing import Optional

from saber.gui.text.segmentation_viewer import HashtagSegmentationViewer
from saber.gui.text.hashtag_manager import HashtagManager
from saber.gui.text.data_manager import TextAnnotationDataManager


class TextAnnotationController:
    """Controller for managing UI state and events in the text annotation window."""
    
    def __init__(self, data_manager: TextAnnotationDataManager, hashtag_manager: HashtagManager):
        self.data_manager = data_manager
        self.hashtag_manager = hashtag_manager
        
        # UI components (set by main window)
        self.segmentation_viewer: Optional[HashtagSegmentationViewer] = None
        self.global_desc_widget = None
        self.seg_desc_widget = None
        self.hashtag_widget = None
        self.image_list = None
        
        # State tracking
        self.current_selected_id: Optional[int] = None
        self._current_run_id: Optional[str] = None
        
        # Setup update timer
        self.color_update_timer = QTimer()
        self.color_update_timer.setSingleShot(True)
        self.color_update_timer.timeout.connect(self.update_segmentation_colors)
    
    def set_ui_components(self, **components):
        """Set UI component references."""
        for name, component in components.items():
            setattr(self, name, component)
    
    def setup_connections(self):
        """Setup signal connections."""
        # List selection
        self.image_list.itemClicked.connect(self.on_image_selected)
        
        # Text changes
        self.global_desc_widget.textChanged.connect(self.on_text_changed)
        self.seg_desc_widget.textChanged.connect(self.on_text_changed)
        
        # Segmentation viewer callbacks
        self.segmentation_viewer.set_selection_callbacks(
            selection_callback=self.on_segmentation_selected_from_viewer,
            deselection_callback=self.on_segmentation_deselected_from_viewer
        )
    
    def get_current_run_id(self) -> str:
        """Get the currently selected run ID."""
        current_row = self.image_list.currentRow()
        return self.data_manager.run_ids[current_row] if current_row >= 0 else None
    
    # Event handlers
    def on_image_selected(self, item):
        """Switch runs with run-level annotation loading and session caching."""
        run_id = item.text()

        # Skip if same run ID
        if hasattr(self, '_current_run_id') and self._current_run_id == run_id:
            return

        prev_run_id = getattr(self, '_current_run_id', None)
        print(f"Switching to run: {run_id}")

        # 1. Save current run's data before switching (if we have a previous run)
        if prev_run_id:
            self.save_current_run_data()
            # Stash current viewer state for session cache
            self.data_manager.stash_session_state(prev_run_id, self.segmentation_viewer)

        # 2. Clear UI state
        self.clear_all_ui_state()

        # 3. Load new run data with session fallback
        self._current_run_id = run_id
        base_image, masks_list, accepted = self.data_manager.read_with_session_fallback(run_id)

        # 4. Fresh viewer state
        self.segmentation_viewer.load_data_fresh(base_image, masks_list)
        # Set accepted masks from session or saved data
        self.segmentation_viewer.set_accepted_indices(accepted)

        # 5. Load run-specific annotations
        self.load_run_annotations(run_id)
        
        print(f"Successfully switched to {run_id}")
    
    def save_current_run_data(self):
        """Save current run's data to memory and zarr."""
        if not self._current_run_id:
            return
        
        print(f"💾 Saving data for current run: {self._current_run_id}")
        
        # Save to memory
        self.save_text_to_memory()
        
        # Save to zarr file
        self.data_manager.save_run_annotations(self._current_run_id, self.hashtag_manager)
    
    def load_run_annotations(self, run_id: str):
        """Load annotations for a specific run."""
        print(f"📂 Loading annotations for run: {run_id}")
        
        # Load run-specific annotations from zarr attributes
        success = self.data_manager.load_run_annotations(run_id, self.hashtag_manager)
        
        if success:
            # Load text into UI
            self.load_text_into_ui(run_id)
            
            # Auto-accept masks that have descriptions (if not already accepted from session)
            self.restore_annotated_masks(run_id)
            
            # Update colors and hashtags
            self.update_hashtags_and_colors_for_run(run_id)
        else:
            print(f"⚠️ No existing annotations found for {run_id}")
            # Clear hashtags for this run since there's no data
            self.hashtag_manager.clear_run_hashtags(run_id)
            self.update_hashtags_and_colors_for_run(run_id)
    
    def load_text_into_ui(self, run_id: str):
        """Load text data into UI widgets."""
        # Disconnect to prevent cascading updates during load
        self.global_desc_widget.textChanged.disconnect()
        self.seg_desc_widget.textChanged.disconnect()
        
        try:
            # Load global description
            global_text = self.data_manager.global_descriptions.get(run_id, "")
            self.global_desc_widget.set_text(global_text)
            
            print(f"📝 Loaded UI text for {run_id}")
            if global_text:
                print(f"   Global: '{global_text}'")
            
        finally:
            # Reconnect signals
            self.global_desc_widget.textChanged.connect(self.on_text_changed)
            self.seg_desc_widget.textChanged.connect(self.on_text_changed)
    
    def restore_annotated_masks(self, run_id: str):
        """Auto-accept masks that have descriptions (if not already accepted from session)."""
        if run_id not in self.data_manager.segmentation_descriptions:
            print(f"No segmentation descriptions found for {run_id}")
            return
        
        descriptions = self.data_manager.segmentation_descriptions[run_id]
        current_accepted = getattr(self.segmentation_viewer, 'accepted_masks', set())
        
        print(f"🎭 Checking {len(descriptions)} masks with descriptions for {run_id}")
        print(f"   Already accepted from session: {len(current_accepted)} masks")
        
        # Only auto-accept masks that aren't already accepted
        newly_accepted = 0
        for seg_id_str, description in descriptions.items():
            try:
                seg_id = int(seg_id_str)
                if seg_id not in current_accepted:
                    self._accept_mask_in_viewer(seg_id)
                    newly_accepted += 1
            except (ValueError, IndexError) as e:
                print(f"   ❌ Error accepting segmentation {seg_id_str}: {e}")
    
    def _accept_mask_in_viewer(self, seg_id: int):
        """Accept a specific mask in the viewer (move from left to right panel)."""
        if seg_id >= len(self.segmentation_viewer.masks):
            print(f"Warning: seg_id {seg_id} >= number of masks {len(self.segmentation_viewer.masks)}")
            return
        
        # Add to accepted masks
        self.segmentation_viewer.accepted_masks.add(seg_id)
        
        # Update accepted stack for undo functionality
        if hasattr(self.segmentation_viewer, 'accepted_stack'):
            if seg_id not in self.segmentation_viewer.accepted_stack:
                self.segmentation_viewer.accepted_stack.append(seg_id)
        
        # Make sure we have the overlay items
        if (hasattr(self.segmentation_viewer, 'left_mask_items') and 
            hasattr(self.segmentation_viewer, 'right_mask_items') and
            seg_id < len(self.segmentation_viewer.left_mask_items) and
            seg_id < len(self.segmentation_viewer.right_mask_items)):
            
            # Hide on left panel, show on right panel
            self.segmentation_viewer.left_mask_items[seg_id].setVisible(False)
            self.segmentation_viewer.right_mask_items[seg_id].setVisible(True)
        else:
            print(f"Warning: overlay items not properly initialized for mask {seg_id}")
    
    def update_hashtags_and_colors_for_run(self, run_id: str):
        """Update hashtags and colors for the current run."""
        print(f"🎨 Updating hashtags and colors for {run_id}")
        
        # Update hashtag UI
        self.hashtag_manager.update_hashtag_list_widget(self.hashtag_widget.get_list_widget())
        
        # Update colors
        self.update_colors_for_run(run_id)
        
        # Debug output
        hashtag_count = sum(1 for hashtag, runs_data in self.hashtag_manager.hashtag_data.items() 
                           if run_id in runs_data)
        print(f"   📊 {hashtag_count} hashtags active for {run_id}")
    
    def clear_all_ui_state(self):
        """Clear all UI state."""
        # Clear text widgets
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

    def on_mask_added(self, mask_index: int):
        """Handle new mask addition."""
        run_id = self.get_current_run_id()
        if not run_id:
            return
        seg_desc = self.data_manager.segmentation_descriptions.setdefault(run_id, {})
        # keys are stored as strings elsewhere
        seg_desc.setdefault(str(mask_index), "")
    
    def select_segmentation(self, segmentation_id: int):
        """Select a segmentation and load its description."""
        # Save current text
        self.save_text_to_memory()
        
        # Clear previous selection
        self.clear_selection_highlight()
        
        # Set new selection
        self.seg_desc_widget.set_selected_segmentation(segmentation_id)
        self.current_selected_id = segmentation_id
        
        # Highlight
        self.add_selection_highlight(segmentation_id)
        
        # Load description for this segmentation
        current_run_id = self.get_current_run_id()
        if (current_run_id in self.data_manager.segmentation_descriptions and 
            str(segmentation_id) in self.data_manager.segmentation_descriptions[current_run_id]):
            existing_text = self.data_manager.segmentation_descriptions[current_run_id][str(segmentation_id)]
            self.seg_desc_widget.set_text(existing_text)
        else:
            self.seg_desc_widget.set_text("")
        
        # Update hashtags for current run
        self.update_hashtags_for_run(current_run_id)
    
    def on_text_changed(self):
        """Handle text changes."""
        self.save_text_to_memory()
        
        # Update colors with short delay
        self.color_update_timer.start(200)
    
    def add_selection_highlight(self, segmentation_id: int):
        """Add a boundary highlight around the selected segmentation."""
        self.segmentation_viewer.highlight_mask(segmentation_id)
    
    def clear_selection_highlight(self):
        """Clear any existing selection highlight."""
        self.segmentation_viewer.clear_highlight()
    
    def update_hashtags_for_run(self, run_id: str):
        """Update hashtags for specific run."""
        # Clear hashtags for this run
        self.hashtag_manager.clear_run_hashtags(run_id)
        
        # Add hashtags from global description
        global_text = self.data_manager.global_descriptions.get(run_id, "")
        if global_text:
            self.hashtag_manager.add_hashtags_from_global(run_id, global_text)
        
        # Add hashtags from segmentation descriptions
        if run_id in self.data_manager.segmentation_descriptions:
            for seg_id, seg_text in self.data_manager.segmentation_descriptions[run_id].items():
                self.hashtag_manager.add_hashtags_from_segmentation(run_id, seg_id, seg_text)
        
        # Update hashtag UI
        self.hashtag_manager.update_hashtag_list_widget(self.hashtag_widget.get_list_widget())
    
    def update_colors_for_run(self, run_id: str):
        """Update segmentation colors for specific run."""
        if run_id not in self.data_manager.segmentation_descriptions:
            return
        
        color_mapping = {}
        for seg_id, description in self.data_manager.segmentation_descriptions[run_id].items():
            hashtags = self.hashtag_manager.extract_hashtags(description)
            if hashtags:
                first_hashtag = sorted(hashtags)[0]
                color_mapping[int(seg_id)] = self.hashtag_manager.get_hashtag_color(first_hashtag)
        
        self.segmentation_viewer.update_mask_colors(color_mapping)
    
    def update_segmentation_colors(self):
        """Update colors for current run."""
        current_run_id = self.get_current_run_id()
        if current_run_id:
            self.update_colors_for_run(current_run_id)
    
    def save_text_to_memory(self):
        """Save current text to memory."""
        current_run_id = self.get_current_run_id()
        if not current_run_id:
            return
        
        global_text = self.global_desc_widget.get_text()
        selected_id = self.seg_desc_widget.get_selected_id()
        seg_text = self.seg_desc_widget.get_text() if selected_id is not None else ""
        
        self.data_manager.save_text_to_memory(current_run_id, global_text, selected_id, seg_text)
    
    # Action methods
    def save_segmentation(self, save_path: str = None):
        """Save segmentation masks and text data for current run."""
        if not save_path:
            print("\nCurrently in viewer mode.\nSave path is not set.")
            return False
        
        current_run_id = self.get_current_run_id()
        if not current_run_id:
            return False
        
        # Save current run's text data
        self.save_text_to_memory()
        
        # Save run-specific annotations
        if not self.data_manager.save_run_annotations(current_run_id, self.hashtag_manager):
            return False
        
        # Save segmentation mask data
        if not self.data_manager.save_masks_data(self.segmentation_viewer, current_run_id):
            return False
        
        return True
    
    def load_next_runID(self, new_row: int):
        """Load the next/previous run ID."""
        new_row = max(0, min(new_row, self.image_list.count() - 1))
        self.image_list.setCurrentRow(new_row)
        self.on_image_selected(self.image_list.item(new_row))