import re
import random
from typing import Set, Dict, Any
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtGui import QColor

class HashtagManager:
    """Manages hashtag extraction, storage, and color assignment."""
    
    def __init__(self):
        self.hashtag_data = {}  # hashtag -> {run_id -> [segmentation_ids]}
        self.hashtag_colors = {}  # hashtag -> color
        self.predefined_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
        ]
    
    def extract_hashtags(self, text: str) -> Set[str]:
        """Extract hashtags from text."""
        hashtag_pattern = r'#\w+'
        hashtags = set(re.findall(hashtag_pattern, text.lower()))
        return hashtags
    
    def get_hashtag_color(self, hashtag: str) -> str:
        """Get or create a unique color for a hashtag."""
        if hashtag not in self.hashtag_colors:
            if len(self.hashtag_colors) < len(self.predefined_colors):
                self.hashtag_colors[hashtag] = self.predefined_colors[len(self.hashtag_colors)]
            else:
                # Generate random color if we run out of predefined ones
                self.hashtag_colors[hashtag] = f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"
        return self.hashtag_colors[hashtag]
    
    def clear_run_hashtags(self, run_id: str):
        """Remove all hashtag data for a specific run."""
        for hashtag in list(self.hashtag_data.keys()):
            if run_id in self.hashtag_data[hashtag]:
                del self.hashtag_data[hashtag][run_id]
            if not self.hashtag_data[hashtag]:  # Remove empty hashtags
                del self.hashtag_data[hashtag]
    
    def add_hashtags_from_global(self, run_id: str, global_text: str):
        """Add hashtags from global description."""
        global_hashtags = self.extract_hashtags(global_text)
        for hashtag in global_hashtags:
            if hashtag not in self.hashtag_data:
                self.hashtag_data[hashtag] = {}
            if run_id not in self.hashtag_data[hashtag]:
                self.hashtag_data[hashtag][run_id] = []
            if 'global' not in self.hashtag_data[hashtag][run_id]:
                self.hashtag_data[hashtag][run_id].append('global')
    
    def add_hashtags_from_segmentation(self, run_id: str, seg_id: str, seg_text: str):
        """Add hashtags from segmentation description."""
        seg_hashtags = self.extract_hashtags(seg_text)
        for hashtag in seg_hashtags:
            if hashtag not in self.hashtag_data:
                self.hashtag_data[hashtag] = {}
            if run_id not in self.hashtag_data[hashtag]:
                self.hashtag_data[hashtag][run_id] = []
            if seg_id not in self.hashtag_data[hashtag][run_id]:
                self.hashtag_data[hashtag][run_id].append(seg_id)
    
    def update_hashtag_list_widget(self, list_widget: QListWidget):
        """Update a QListWidget with current hashtags."""
        list_widget.clear()
        
        if not self.hashtag_data:
            item = QListWidgetItem("No hashtags")
            item.setForeground(QColor("gray"))
            list_widget.addItem(item)
            return
        
        # Add hashtags to the list with their colors
        for hashtag in sorted(self.hashtag_data.keys()):
            item = QListWidgetItem(hashtag)
            color = self.get_hashtag_color(hashtag)
            item.setForeground(QColor(color))
            list_widget.addItem(item)
    
    def get_data_for_save(self) -> Dict[str, Any]:
        """Get hashtag data for saving."""
        return {
            'hashtag_data': self.hashtag_data,
            'hashtag_colors': self.hashtag_colors
        }
    
    def load_data_from_save(self, data: Dict[str, Any]):
        """Load hashtag data from saved data."""
        self.hashtag_data = data.get('hashtag_data', {})
        self.hashtag_colors = data.get('hashtag_colors', {})