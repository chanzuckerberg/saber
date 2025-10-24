from typing import Dict, Any, Optional, Mapping
import zarr, threading, json
import numpy as np

# -----------------------------
# JSON-safe conversion
# -----------------------------
def _to_jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    # Fallback: store string repr to avoid serialization errors
    return str(obj)

# Global zarr writer instance (initialized once)
_zarr_writer = None
_writer_lock = threading.Lock()

class ParallelZarrWriter:
    """
    Thread-safe zarr writer that handles incremental writes as runs complete.
    Designed for large-scale processing (500-2000+ runs).
    """
    
    def __init__(self, zarr_path: str):
        """
        Initialize the zarr writer.
        
        Args:
            zarr_path: Path to the zarr file
        """

        # Use zarr's built-in thread synchronization
        self.zarr_path = zarr_path
        synchronizer = zarr.ThreadSynchronizer()
        
        # Open zarr store with synchronization and VCP-compatible settings
        self.store = zarr.NestedDirectoryStore(zarr_path)
        self.store.dimension_separator = '/'
        self.zroot = zarr.open_group(
            store=self.store, 
            mode='w',
            synchronizer=synchronizer,
        )
        
        # Thread-safe counter for run indexing
        self._run_counter = 0
        self._lock = threading.Lock()
        
        print(f"Initialized zarr store at: {zarr_path}")

    # -----------------------------
    # Attr helpers
    # -----------------------------
    def set_dict_attr(
        self,
        key: str,
        data: Mapping[str, Any],
        *,
        merge_missing: bool = False
    ) -> None:
        """
        Write a dictionary to root attributes under `key`.

        - If merge_missing=False: overwrite entirely.
        - If merge_missing=True: only fill in keys that are absent.

        Ensures values are JSON-serializable (numpy -> Python types).
        """
        safe = _to_jsonable(dict(data))
        with self._lock:
            if merge_missing:
                existing = self.zroot.attrs.get(key)
                if isinstance(existing, dict):
                    merged = dict(existing)
                    changed = False
                    for k, v in safe.items():
                        if k not in merged:
                            merged[k] = v
                            changed = True
                    if changed:
                        self.zroot.attrs[key] = merged
                    return
            # default: overwrite
            self.zroot.attrs[key] = safe        
    
    def get_next_run_index(self) -> int:
        """Get the next available run index in a thread-safe manner."""
        with self._lock:
            run_index = self._run_counter
            self._run_counter += 1
            return run_index
    
    def write(
        self, run_name: str, image: np.ndarray, masks: np.ndarray, 
        pixel_size: float = None,
        metadata: Dict[str, Any] = None) -> int:
        """
        Write data for a single run to the zarr file.
        This is thread-safe and can be called from multiple GPUs simultaneously.
        
        Args:
            run_name: Name of the run
            image: Image data array
            masks: Mask data array  
            metadata: Optional metadata dictionary
            
        Returns:
            run_index: The index assigned to this run
        """
        
        # Default pixel size to 1.0 if not provided
        if pixel_size is None:
            pixel_size = 1.0

        # Get thread-safe run index
        run_index = self.get_next_run_index()
        
        try:
            # Create group for this run - zarr handles concurrent group creation
            run_group = self.zroot.create_group(run_name)   
            
            # Store run metadata
            if metadata:
                for key, value in metadata.items():
                    run_group.attrs[key] = value
            
            # Write image dataset
            run_group.create_dataset(
                "0", 
                data=image, 
                dtype=image.dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=2, shuffle=2),
            )
            # Add VCP attributes to the run group
            add_attributes(run_group, pixel_size)            
            
            # Write masks dataset
            labels_group = run_group.create_group("labels")
            labels_group.create_dataset(
                "0",
                data=masks,
                dtype=masks.dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=2, shuffle=2),
            )
            add_attributes(labels_group, pixel_size, True)
            
            # print(f"✅ Written {run_name} to {self.zarr_path}")
            return run_index
            
        except Exception as e:
            print(f"❌ Error writing {run_name} to zarr: {str(e)}")
            raise
    
    def finalize(self):
        """Finalize the zarr file and add global metadata."""
        try:
            # Add global metadata
            self.zroot.attrs['total_runs'] = self._run_counter
            self.zroot.attrs['creation_complete'] = True
            
            # Ensure all data is flushed
            self.store.close()
            print(f"📁 Zarr file finalized with {self._run_counter} runs")
            
        except Exception as e:
            print(f"⚠️  Error finalizing zarr file: {str(e)}")

def get_zarr_writer(zarr_path: str) -> ParallelZarrWriter:
    """Get or create the global zarr writer instance."""
    global _zarr_writer
    
    with _writer_lock:
        if _zarr_writer is None:
            _zarr_writer = ParallelZarrWriter(zarr_path)
        return _zarr_writer
    
def add_attributes(
    zarr_group: zarr.Group,
    voxel_size: float = 1.0, 
    is_3d: bool = False,
    voxel_size_z: float = 1.0) -> None:
    """
    Add VCP-compatible multiscale attributes to any zarr group.
    
    :param zarr_group: Any zarr group (run group, labels group, etc.)
    :param voxel_size: Voxel size in nanometers
    :param voxel_size_z: Voxel size in nanometers for the z-axis
    :param is_3d: Whether this is 3D data (z,y,x) or 2D data (y,x)
    """
    if is_3d:
        axes = [
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"}
        ]
        scale = [voxel_size_z, voxel_size, voxel_size]
    else:
        axes = [
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"}
        ]
        scale = [voxel_size, voxel_size]
    
    zarr_group.attrs.update({
        "multiscales": [
            {
                "axes": axes,
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {
                                "scale": scale,
                                "type": "scale"
                            }
                        ],
                        "path": "0"
                    }
                ],
                "name": "/",
                "version": "0.4"
            }
        ]
    })