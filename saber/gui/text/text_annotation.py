from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QComboBox, QPushButton, QListWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

class GlobalDescriptionWidget(QWidget):
    """Widget for global description input."""
    
    textChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label = QLabel("Global Description:")
        label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(label)
        
        # Text edit
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter a global description for this tomogram/image... Use #hashtags to categorize")
        self.text_edit.setMaximumHeight(50)
        self.text_edit.textChanged.connect(self.textChanged.emit)
        layout.addWidget(self.text_edit)
    
    def get_text(self) -> str:
        return self.text_edit.toPlainText()
    
    def set_text(self, text: str):
        self.text_edit.setPlainText(text)

class SegmentationDescriptionWidget(QWidget):
    """Widget for per-segmentation description input."""
    
    textChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.selected_segmentation_id = None
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label = QLabel("Selected Segmentation Description:")
        label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 2px;")
        layout.addWidget(label)
        
        # Status label
        self.status_label = QLabel("No segmentation selected")
        self.status_label.setStyleSheet("color: gray; font-size: 10px; margin-bottom: 2px;")
        layout.addWidget(self.status_label)
        
        # Text edit
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Click on a segmentation mask, then enter description with #hashtags...")
        self.text_edit.setMaximumHeight(50)
        self.text_edit.setEnabled(False)
        self.text_edit.textChanged.connect(self.textChanged.emit)
        layout.addWidget(self.text_edit)
    
    def set_selected_segmentation(self, segmentation_id: int):
        """Set the selected segmentation ID."""
        self.selected_segmentation_id = segmentation_id
        self.status_label.setText(f"Selected: Segmentation {segmentation_id}")
        self.text_edit.setEnabled(True)
    
    def clear_selection(self):
        """Clear the segmentation selection."""
        self.selected_segmentation_id = None
        self.status_label.setText("No segmentation selected")
        self.text_edit.setEnabled(False)
        self.text_edit.setPlainText("")
    
    def get_text(self) -> str:
        return self.text_edit.toPlainText()
    
    def set_text(self, text: str):
        self.text_edit.setPlainText(text)
    
    def get_selected_id(self):
        return self.selected_segmentation_id


class ControlPanelWidget(QWidget):
    """Widget for action buttons (simplified for hashtag-based workflow)."""
    
    saveClicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)
        
        # Add some spacing before the save button
        layout.addStretch()
        
        # Save button
        self.save_button = QPushButton("Save Segmentation & Text")
        self.save_button.clicked.connect(self.saveClicked.emit)
        layout.addWidget(self.save_button)
        
        # Add some spacing after the save button
        layout.addStretch()

class HashtagListWidget(QWidget):
    """Widget for displaying hashtags."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(250)
        layout.addWidget(self.list_widget)
    
    def get_list_widget(self) -> QListWidget:
        """Get the internal list widget for hashtag manager updates."""
        return self.list_widget