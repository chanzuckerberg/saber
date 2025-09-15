import saber.finetune.helper as helper
from saber.utils import preprocessing 
from torch.utils.data import Dataset
import numpy as np
import zarr, torch

class AutoMaskDataset(Dataset):
    def __init__(self, 
                 tomogram_zarr_path: str = None, 
                 fib_zarr_path: str = None,
                 transform = None,
                 slabs_per_volume_per_epoch: int = 10,
                 slices_per_fib_per_epoch: int = 5,
                 slab_thickness: int = 5):
        """
        Args:
            tomogram_zarr_path: Path to the tomogram zarr store
            fib_zarr_path: Path to the fib zarr store
            transform: Transform to apply to the data
            slabs_per_volume_per_epoch: Number of slabs per volume per epoch
            slices_per_fib_per_epoch: Number of slices per fib per epoch
            slab_thickness: Thickness of the slab
        """

        # Slabs per Epoch 
        self.slab_thickness = slab_thickness
        self.slabs_per_volume_per_epoch = slabs_per_volume_per_epoch
        self.slices_per_fib_per_epoch = slices_per_fib_per_epoch
        
        # Grid and Positive Points for AutoMaskGenerator
        self.points_per_side = 32
        self.points_per_batch = 64
        self.min_area = 0.001
        self.k_pos = 2

        # Check if both data types are available
        if tomogram_zarr_path is None and fib_zarr_path is None:
            raise ValueError("At least one of tomogram_zarr_path or fib_zarr_path must be provided")        

        # Flags to track which data types are available
        self.has_tomogram = tomogram_zarr_path is not None
        self.has_fib = fib_zarr_path is not None

        # Initialize tomogram data if provided
        if self.has_tomogram:
            self.tomogram_store = zarr.open(tomogram_zarr_path, mode='r')
            self.tomogram_keys = [k for k in self.tomogram_store.keys() if not k.startswith('.')]
            self.n_tomogram_volumes = len(self.tomogram_keys)
            self.tomo_shapes = {}
            for i, key in enumerate(self.tomogram_keys):
                self.tomo_shapes[i] = self.tomogram_store[key]['0'].shape
        else:
            self.n_tomogram_volumes = 0
            self.tomo_shapes = {}
            self.tomogram_keys = []
            
        # Initialize fib data if provided
        if self.has_fib:
            self.fib_store = zarr.open(fib_zarr_path, mode='r')
            self.fib_keys = [k for k in self.fib_store.keys() if not k.startswith('.')]
            self.n_fib_volumes = len(self.fib_keys)
            self.fib_shapes = {}
            for i, key in enumerate(self.fib_keys):
                self.fib_shapes[i] = self.fib_store[key]['0'].shape
        else:
            self.n_fib_volumes = 0
            self.fib_shapes = {}
            self.fib_keys = []

        # Resample epoch
        self.resample_epoch()

        # Verbose Flag
        self.verbose = False
    
    def resample_epoch(self):
        """ Generate new random samples for this epoch """
        self.tomogram_samples = []
        self.fib_samples = []
        
        # Sample random slabs from each tomogram
        if self.has_tomogram:
            for vol_idx in range(self.n_tomogram_volumes):
                volume_shape = self.tomo_shapes[vol_idx]
                # Valid range for center of slab
                valid_z_min = int(volume_shape[0] / 4)
                valid_z_max = int(volume_shape[0] * (3 / 4))
                
                if valid_z_max > valid_z_min:
                    z_positions = np.random.randint(
                        valid_z_min, 
                        valid_z_max, 
                        size=self.slabs_per_volume_per_epoch
                    )
                    
                    for z_pos in z_positions:
                        self.tomogram_samples.append((vol_idx, z_pos))
            np.random.shuffle(self.tomogram_samples) # Shuffle samples
        
        # Sample random slices from each FIB volume
        if self.has_fib:
            for fib_idx in range(self.n_fib_volumes):
                fib_shape = self.fib_shapes[fib_idx]
                # Sample random z positions from this FIB volume
                z_positions = np.random.randint(
                    0, 
                    fib_shape[0], 
                    size=self.slices_per_fib_per_epoch
                )
                
                for z_pos in z_positions:
                    self.fib_samples.append((fib_idx, z_pos))        
            np.random.shuffle(self.fib_samples) # Shuffle samples
        
        # Set epoch length
        self.epoch_length = len(self.tomogram_samples) + len(self.fib_samples)
    
    def __len__(self):
        return self.epoch_length
    
    def __getitem__(self, idx):

        # Get item from tomogram or FIB
        if idx < len(self.tomogram_samples) and self.has_tomogram:
            return self._get_tomogram_item(idx)
        else:
            return self._get_fib_item(idx - len(self.tomogram_samples))

    def _get_tomogram_item(self, idx):

        # Randomly select a tomogram volume
        vol_idx, z_pos = self.tomogram_samples[idx]
        key = self.tomogram_keys[vol_idx]

        # Load image and segmentation slab
        z_start = z_pos - self.slab_thickness // 2
        z_end = z_pos + self.slab_thickness // 2 + 1
        image_slab = self.tomogram_store[key]['0'][z_start:z_end]
        seg_slab = self.tomogram_store[key]['labels/0'][z_start:z_end]
        
        # Project slab and normalize 
        image_projection = preprocessing.project_tomogram(image_slab)
        image_2d = preprocessing.proprocess(image_projection)          # 3xHxW
        
        # Project segmentation  
        seg_2d = preprocessing.project_segmentation(seg_slab)  # HxW
        
        return self._package_image_item(image_2d, seg_2d)
    
    def _get_fib_item(self, idx):

        # Randomly select a FIB volume
        fib_idx, z_pos = self.fib_samples[idx]
        key = self.fib_keys[fib_idx]
        
        # Load FIB image and segmentation
        image = self.fib_store[key]['0'][z_pos,]
        image_2d = preprocessing.proprocess(image)
        seg_2d = self.fib_store[key]['labels/0'][z_pos,]
        
        return self._package_image_item(image_2d, seg_2d)

    def _gen_grid_points(self, h: int, w: int) -> np.ndarray:
        """
        Generate grid points for a given image size
        """
        xs = np.linspace(0.5, w - 0.5, self.points_per_side, dtype=np.float32)
        ys = np.linspace(0.5, h - 0.5, self.points_per_side, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        return np.stack([xx.ravel(), yy.ravel()], axis=1)  # (G,2) as (x,y)

    def _package_image_item(self, 
        image_2d: np.ndarray, 
        segmentation: np.ndarray):
        """
        Build per-component targets using grid-hit instances.
        - Splits each hit instance into connected components.
        - Drops components smaller than min_area_frac * (H*W).
        - Emits only positive clicks + boxes (no negatives).
        Returns:
            {
                "image": HxWx3 uint8,
                "masks":  list[H x W] float32 in {0,1},
                "points": list[#p x 2] float32 (xy),
                "labels": list[#p] float32 (all ones),
                "boxes":  list[4] float32 (x0,y0,x1,y1)
            }
        """

        h, w = segmentation.shape
        min_pixels = 0
        # min_pixels = int(self.min_area * h * w)

        # which instances to train on for this image
        grid_points = self._gen_grid_points(h, w)
        inst_ids = helper.instances_from_grid(grid_points, segmentation)
        masks_t, points_t, labels_t, boxes_t = [], [], [], []

        for iid in inst_ids:
            comps = helper.components_for_id(segmentation, iid, min_pixels)
            for comp in comps:
                # box from this component
                box = helper.mask_to_box(comp)
                if box is None:
                    continue

                # sample clicks from this component (NOT the full instance)
                pts = helper.sample_positive_points(comp, k=self.k_pos)
                if pts.shape[0] == 0:
                    continue

                masks_t.append(torch.from_numpy(comp.astype(np.float32)))
                points_t.append(torch.from_numpy(pts.astype(np.float32)))
                labels_t.append(torch.from_numpy(np.ones((pts.shape[0],), dtype=np.float32)))
                boxes_t.append(torch.from_numpy(box.astype(np.float32)))

        # fallback to a harmless dummy if nothing was hit by the grid (keeps loader stable)
        if len(masks_t) == 0:
            print("No masks found")
            masks_t = [torch.from_numpy(np.zeros_like(segmentation, dtype=np.float32))]
            points_t = [torch.from_numpy(np.zeros((1, 2), dtype=np.float32))]
            labels_t = [torch.from_numpy(np.ones((1,), dtype=np.float32))]
            boxes_t = [torch.from_numpy(np.array([0, 0, 1, 1], dtype=np.float32))]

        # Apply transforms
        if self.transform:
            data = self.transform({'image': image_2d, 'masks': masks_t})
            image_2d, masks_t = data['image'], data['masks']

        return {
            "image": image_2d,     # HxWx3 uint8
            "masks": masks_t,     # list[H x W] float32 in {0,1}
            "points": points_t,   # list[#p x 2] float32 (xy)
            "labels": labels_t,   # list[#p] all ones
            "boxes": boxes_t,     # list[4] (x0,y0,x1,y1)
        }