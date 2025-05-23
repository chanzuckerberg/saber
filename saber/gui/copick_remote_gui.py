from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QListWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
from saber.gui.segmentation_picker import SegmentationViewer  
import saber.utilities as utils
import numpy as np
import copick, sys, click

class MainWindow(QMainWindow):
    def __init__(self, config, tomo_info, seg_info, slab_thickness):
        super().__init__()

        # Load Copick Project
        self.root = copick.from_file(config)
        self.run_ids = [run.name for run in self.root.runs]

        # Parse Input Parameters
        self.tomo_algorithm, self.voxel_size = tomo_info.split(',')
        self.segUserID, self.segSessionID, self.segName = seg_info.split(',')
        self.slab_thickness = slab_thickness

        self.setWindowTitle("Two-Image Viewer with Image List")

        # Create the splitter for left (list) and right (viewer)
        self.splitter = QSplitter(Qt.Horizontal, self)

        # --- Left Panel: RunIDs List ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        self.image_list = QListWidget()
        for image_name in self.run_ids:
            self.image_list.addItem(image_name)

        # Highlight the first entry by default
        if self.image_list.count() > 0:  # Ensure the list is not empty
            self.image_list.setCurrentRow(0)

        self.left_layout.addWidget(self.image_list)
        self.splitter.addWidget(self.left_panel)

        # --- Right Panel: Segmentation Viewer ---
        # Read data for the first run
        (initial_image, initial_masks) = self.read_data(self.run_ids[0])
  
        self.segmentation_viewer = SegmentationViewer(initial_image, initial_masks)
        self.splitter.addWidget(self.segmentation_viewer)

        # Set splitter as the central widget of the main window
        self.setCentralWidget(self.splitter)

        # Initial size
        self.splitter.setSizes([175, 850])  # Set left panel to 150px and right panel to 850px        
        self.resize(1000, 600)

        # Connect signals
        self.image_list.itemClicked.connect(self.on_image_selected)

    def read_data(self, run_id):
        """
        Read the base image (tomogram) and segmentation masks for a given run ID.
        :param run_id: The ID of the selected run
        :return: base_image (numpy array), masks (list of numpy arrays)
        """

        # Get Run
        run = self.root.get_run(run_id)

        # Get Tomogram
        vol = run.get_voxel_spacing(self.voxel_size).get_tomograms(self.tomo_algorithm)
        vol = vol[0].numpy()

        # Convert tomogram to a 2D slice (e.g., a middle slice) for visualization
        mid_z = vol.shape[0] // 2  # Middle slice along the z-axis
        base_image = utils.project_tomogram(vol, deltaZ=self.slab_thickness)        

        # Get Segmentation
        segmentations = utils.get_segmentation_array(
            run, 
            voxel_spacing=self.voxel_size, 
            segmentation_name=self.segName, 
            user_id=self.segUserID)
        
        # Convert segmentations to a list of 2D masks (assuming same mid-z slice)
        segmentation_classes = np.unique(segmentations)
        segmentation_classes = segmentation_classes[segmentation_classes != 0]  # Exclude class 0

        # Create a binary mask for each segmentation class
        masks = [
            (segmentations[mid_z, :, :] == seg_class).astype(np.uint8)
            for seg_class in segmentation_classes
        ]   

        return base_image, masks

    def on_image_selected(self, item):
        """
        Load the selected image into the viewer.
        :param item: The selected QListWidgetItem
        """
        run_id = item.text()  # Get the selected run ID
        print(f"Selected run ID: {run_id}")

        # Read the data for the selected run ID
        try:
            base_image, masks = self.read_data(run_id)
        except Exception as e:
            print(f"Error loading data for run ID {run_id}: {e}")
            return

        # Load the data into the segmentation viewer
        self.segmentation_viewer.load_data(base_image, masks)

@click.command(context_settings={"show_default": True})
@click.option('--config', type=str, required=True, help="Path to the configuration file.")
def main():
    # Load image list
    config = '/hpc/projects/group.czii/krios1.processing/copick/24jul29c/run001/copick_config.json'
    # config = '/Users/jonathan.schwartz/Documents/copick_bruno/lysosomes/24jul29c.json'
    tomo_info = 'denoised,10.0' # tomogram_algorithm,voxel_size
    seg_info = 'cryoSAM2,1,organelles' # userID,sessionID,name
    slab_thickness = 10

    # Start the app
    app = QApplication(sys.argv)
    main_window = MainWindow(config=config, 
                             tomo_info=tomo_info, 
                             seg_info=seg_info, 
                             slab_thickness=slab_thickness)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
