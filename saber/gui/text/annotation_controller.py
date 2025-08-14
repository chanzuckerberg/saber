"""
Controller for SAM2-ET text annotation UI.
Handles UI state, event handling, and coordination between components.
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
        base_image, masks = self.data_manager.read_data(run_id)
        
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
            global_text = self.data_manager.global_descriptions.get(run_id, "")
            self.global_desc_widget.set_text(global_text)
            
            # Load previously annotated segmentations and restore them
            self.restore_previously_annotated_masks(run_id)
            
            # Update hashtags for this run only
            self.update_hashtags_for_run(run_id)
            
            # Update colors based on existing descriptions for this run
            self.update_colors_for_run(run_id)
            
        finally:
            # Reconnect signals
            self.global_desc_widget.textChanged.connect(self.on_text_changed)
            self.seg_desc_widget.textChanged.connect(self.on_text_changed)
    
    def restore_previously_annotated_masks(self, run_id: str):
        """Restore previously annotated masks to the right panel with hashtag colors."""
        # Load saved mask data
        saved_masks_data = self.data_manager.load_masks_with_descriptions(run_id)
        
        if not saved_masks_data:
            print(f"No previously annotated masks found for {run_id}")
            return
        
        print(f"Restoring {len(saved_masks_data)} previously annotated masks for {run_id}")
        
        # Clear current accepted masks
        self.segmentation_viewer.accepted_masks.clear()
        
        # Restore each previously annotated mask
        for seg_key, mask_data in saved_masks_data.items():
            seg_id = mask_data['segmentation_id']
            
            # Add to accepted masks
            self.segmentation_viewer.accepted_masks.add(seg_id)
            
            # Make sure the right panel overlay is visible
            if (hasattr(self.segmentation_viewer, 'right_mask_items') and 
                seg_id < len(self.segmentation_viewer.right_mask_items)):
                self.segmentation_viewer.right_mask_items[seg_id].setVisible(True)
            
            print(f"Restored segmentation {seg_id}: '{mask_data['description']}'")
        
        # Update the accepted stack (for undo functionality)
        if hasattr(self.segmentation_viewer, 'accepted_stack'):
            self.segmentation_viewer.accepted_stack = list(self.segmentation_viewer.accepted_masks)
        
        print(f"Successfully restored {len(saved_masks_data)} annotated masks for {run_id}")
    
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
        if (current_run_id in self.data_manager.segmentation_descriptions and 
            str(segmentation_id) in self.data_manager.segmentation_descriptions[current_run_id]):
            existing_text = self.data_manager.segmentation_descriptions[current_run_id][str(segmentation_id)]
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
    
    def update_hashtags_for_run(self, run_id: str):
        """Update hashtags for specific run only."""
        # Clear hashtags for this run only
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
        """Simple color update for current run."""
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
        """Save segmentation masks and text data."""
        if not save_path:
            print("\nCurrently in viewer mode.\nSave path is not set.")
            return False
        
        # Save text data first
        self.save_text_to_memory()
        if not self.data_manager.save_text_data(self.hashtag_manager):
            return False
        
        # Save segmentation data
        current_run_id = self.get_current_run_id()
        if not self.data_manager.save_masks_data(self.segmentation_viewer, current_run_id):
            return False
        
        return True
    
    def load_next_runID(self, new_row: int):
        """Load the next/previous run ID."""
        new_row = max(0, min(new_row, self.image_list.count() - 1))
        self.image_list.setCurrentRow(new_row)
        self.on_image_selected(self.image_list.item(new_row))
