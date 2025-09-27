from scipy.ndimage import binary_erosion, binary_dilation
from monai.transforms import Compose, EnsureChannelFirstd
import saber.finetune.helper as helper
from saber.utils import preprocessing
from torch.utils.data import Dataset
import zarr, torch, random
from tqdm import tqdm
import numpy as np

class AutoMaskDataset(Dataset):
    def __init__(self, 
                 tomogram_zarr_path: str = None, 
                 fib_zarr_path: str = None,
                 transform = None,
                 slabs_per_volume_per_epoch: int = 10,
                 slices_per_fib_per_epoch: int = 5,
                 slab_thickness: int = 5,
                 seed: int = 42):
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
        self.min_area = 0.001
        self.k_min = 50
        self.k_max = 100
        self.transform = transform
        self.keep_fraction = 0.5

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
            self.tomo_shapes = {}
            for i, key in tqdm(enumerate(self.tomogram_keys), 
                               total=len(self.tomogram_keys), desc="Estimating zrange for tomograms"):
                try: 
                    self.tomo_shapes[i] = self._estimate_zrange(key)
                except Exception as e:
                    print(f"Error estimating zrange for tomogram {key}: {e}")
                    # remove key from tomogram_keys
                    self.tomogram_keys.remove(key)
            self.n_tomogram_volumes = len(self.tomogram_keys)
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
        
        # Random seed
        self.seed = seed
        self._rng = np.random.RandomState(seed)


        # Verbose Flag
        self.verbose = False

        # Samples
        self.tomogram_samples = []
        self.fib_samples = []
        self._prev_tomogram_samples = []
        self._prev_fib_samples = []

        # First sampling epoch
        self.resample_epoch()

    def _estimate_zrange(self, key, band=(0.3, 0.7), threshold=0):
        """
        Returns (z_min, z_max) inclusive bounds for valid slab centers
        where there is some foreground in the labels.
        - threshold: min # of fg pixels to count a slice as non-empty (0 = any)
        - band: fraction of Z to consider (lo, hi)
        """

        nz = self.tomogram_store[key]['0'].shape[0]
        min_offset, max_offset = int(nz * band[0]), int(nz * band[1])
        mask = self.tomogram_store[key]['labels/0'][min_offset:max_offset,]
        vals = mask.sum(axis=(1,2))
        vals = np.nonzero(vals)[0]
        max_val = vals.max() - self.slab_thickness + min_offset
        min_val = vals.min() + self.slab_thickness + min_offset
        return int(min_val), int(max_val)
    
    def resample_epoch(self):
        """ Generate new random samples for this epoch """
        
        # Sample random slabs from each tomogram
        if self.has_tomogram:
            print(f"Re-Sampling {self.slabs_per_volume_per_epoch} slabs from {self.n_tomogram_volumes} tomograms")
            new_tomo_samples = []
            for vol_idx in range(self.n_tomogram_volumes):

                # Sample random z positions from this tomogram volume
                z_min, z_max = self.tomo_shapes[vol_idx]
                z_positions = self._rng.randint(
                    z_min, 
                    z_max, 
                    size=self.slabs_per_volume_per_epoch
                )
                # Add to samples
                for z_pos in z_positions:
                    new_tomo_samples.append((vol_idx, z_pos))

            self.tomogram_samples = self._update_samples(self.tomogram_samples, new_tomo_samples)

            # Shuffle samples
            self._rng.shuffle(self.tomogram_samples) 
        
        # Sample random slices from each FIB volume
        if self.has_fib:
            print(f"Re-Sampling {self.slices_per_fib_per_epoch} slices from {self.n_fib_volumes} FIB volumes")
            new_fib_samples = []
            for fib_idx in range(self.n_fib_volumes):
                fib_shape = self.fib_shapes[fib_idx]
                # Sample random z positions from this FIB volume
                z_positions = self._rng.randint(
                    0, 
                    fib_shape[0], 
                    size=self.slices_per_fib_per_epoch
                )
                
                for z_pos in z_positions:
                    new_fib_samples.append((fib_idx, z_pos))        

            self.fib_samples = self._update_samples(self.fib_samples, new_fib_samples)
            self._rng.shuffle(self.fib_samples) # Shuffle samples
        
        # Set epoch length
        self.epoch_length = len(self.tomogram_samples) + len(self.fib_samples)
    
    def _update_samples(self, old, new):
        """
        Return a mixed list with size == len(new):
        - keep = min(round(len(new)*keep_fraction), len(old)) from 'old'
        - add  = len(new) - keep from 'new'
        """
        target = len(new)
        if target == 0:
            return []

        # choose keep set from *old* list, new set from *new* list
        keep = min(int(round(target * self.keep_fraction)), len(old))
        add  = target - keep

        # keep set from *old* list
        if keep > 0:
            keep_idx = self._rng.choice(len(old), size=keep, replace=False)
            kept = [old[i] for i in keep_idx]
        else:
            kept = []

        # add set from *new* list
        if add > 0:
            new_idx = self._rng.choice(len(new), size=add, replace=False)
            added = [new[i] for i in new_idx]
        else:
            added = []

        # return mixed list
        return kept + added

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
        
        # Project slab and segmentation 
        image_2d = preprocessing.project_tomogram(image_slab)
        seg_2d = preprocessing.project_segmentation(seg_slab)  # HxW
        
        return self._package_image_item(image_2d, seg_2d)
    
    def _get_fib_item(self, idx):

        # Randomly select a FIB volume
        fib_idx, z_pos = self.fib_samples[idx]
        key = self.fib_keys[fib_idx]
        
        # Load FIB image and segmentation
        image = self.fib_store[key]['0'][z_pos,].astype(np.float32)
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

    def _sample_negative_ring(self, comp, other_inst=None, ring=3, max_neg=16, shape=None):
        h, w = shape
        comp_b = comp.astype(np.bool_)
        outer = binary_dilation(comp_b, iterations=ring) & (~comp_b)
        if other_inst is not None:
            # avoid putting negatives inside any other instance
            outer = outer & (~other_inst.astype(np.bool_))
        ys, xs = np.where(outer)
        if len(xs) == 0:
            return np.zeros((0, 2), np.float32)
        k = min(max_neg, len(xs))
        idx = self._rng.choice(len(xs), size=k, replace=False)
        return np.stack([xs[idx].astype(np.float32), ys[idx].astype(np.float32)], axis=1)

    def _sample_points_in_mask(
        self,
        comp: np.ndarray,
        grid_points: np.ndarray,
        shape: tuple[int, int],
        jitter_px: float = 1.0,
        k_cap: int = 300,
        boundary_frac: float = 0.35,
    ) -> np.ndarray:
        """
        Pick informative clicks from a dense grid:
        - favor boundary points
        - cap count ~sqrt(area) up to k_cap
        Returns float32 array [K,2] in (x,y).
        """
        h, w = shape
        if comp.sum() == 0:
            return np.zeros((0, 2), np.float32)

        # ----- cast to boolean for morphology -----
        comp_b = comp.astype(np.bool_)             # important: avoid TypeError with '^'

        # grid → nearest pixel indices for inside/boundary tests
        gx = np.clip(np.rint(grid_points[:, 0]).astype(int), 0, w - 1)
        gy = np.clip(np.rint(grid_points[:, 1]).astype(int), 0, h - 1)

        inside = comp_b[gy, gx]
        cand = grid_points[inside]
        if cand.shape[0] == 0:
            return np.zeros((0, 2), np.float32)

        # ----- boundary mask (inner ring) -----
        eroded = binary_erosion(comp_b, iterations=2)
        boundary_b = np.logical_and(comp_b, np.logical_not(eroded))  # same as comp ^ eroded on booleans

        on_b = boundary_b[gy, gx] & inside
        cand_b = grid_points[on_b]
        cand_i = grid_points[inside & (~on_b)]

        # target k ~ c * area but capped
        area = float(comp_b.sum())
        k_target = int(
            min( k_cap, max(24, area * 0.12) )
        )

        kb = int(boundary_frac * k_target)
        ki = k_target - kb

        rng = self._rng
        take_b = min(kb, len(cand_b))
        take_i = min(ki, len(cand_i))

        # if boundary is too small, backfill from interior
        if take_b + take_i == 0:
            return np.zeros((0, 2), np.float32)

        if take_b:
            idx_b = rng.choice(len(cand_b), size=take_b, replace=False)
            pts_b = cand_b[idx_b]
        else:
            pts_b = np.zeros((0, 2), np.float32)

        if take_i:
            idx_i = rng.choice(len(cand_i), size=take_i, replace=False)
            pts_i = cand_i[idx_i]
        else:
            pts_i = np.zeros((0, 2), np.float32)

        pts = np.concatenate([pts_b, pts_i], axis=0).astype(np.float32)

        # jitter a touch to avoid perfect grid regularity
        if jitter_px > 0 and pts.shape[0] > 0:
            jitter = rng.uniform(-jitter_px, jitter_px, size=pts.shape).astype(np.float32)
            pts += jitter
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        return pts   

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
                "image": HxW,
                "masks":  list[H x W] float32 in {0,1},
                "points": list[#p x 2] float32 (xy),
                "labels": list[#p] float32 (all ones),
                "boxes":  list[4] float32 (x0,y0,x1,y1)
            }
        """
        
        # Apply transforms to image and segmentation
        if self.transform:
            sample = self.transform({'image': image_2d, 'mask': segmentation})
            image_2d, segmentation = sample['image'], sample['mask']      

        # Get image and segmentation shapes
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
                # pts = helper.sample_positive_points(comp, k=self.k_pos)
                pts = self._sample_points_in_mask(comp, grid_points, shape=(h, w))
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

        # Normalize the Image
        image_2d = preprocessing.proprocess(image_2d)          # 3xHxW

        return {
            "image": image_2d,     # HxWx3
            "masks": masks_t,     # list[H x W] float32 in {0,1}
            "points": points_t,   # list[#p x 2] float32 (xy)
            "labels": labels_t,   # list[#p] all ones
            "boxes": boxes_t,     # list[4] (x0,y0,x1,y1)
        }