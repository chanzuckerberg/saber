from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QListWidget, QPushButton
    )

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

class MaskSelector(QWidget):
    def __init__(self):
        super().__init__()

        # Main Layout
        layout = QVBoxLayout(self)

        # Dropdown Menu and Label in a Horizontal Layout
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(QLabel("Select a class:"))
        
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Class 1", "Class 2", "Class 3"])  # Add class options
        horizontal_layout.addWidget(self.dropdown)

        # Add the horizontal layout to the main layout
        layout.addLayout(horizontal_layout)

        # Set the main layout for the widget
        self.setLayout(layout)

    def get_selected_class(self):
        """Return the currently selected class from the dropdown."""
        return self.dropdown.currentText()