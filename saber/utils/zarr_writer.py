from typing import Dict, Any
import zarr, threading
import numpy as np

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
            synchronizer: Optional zarr synchronizer for distributed writing
        """

        # Use zarr's built-in thread synchronization
        self.zarr_path = zarr_path
        synchronizer = zarr.ThreadSynchronizer()
        
        # Open zarr store with synchronization
        self.store = zarr.NestedDirectoryStore(zarr_path)
        self.zroot = zarr.open_group(
            store=self.store, 
            mode='w',
            synchronizer=synchronizer
        )
        
        # Thread-safe counter for run indexing
        self._run_counter = 0
        self._counter_lock = threading.Lock()
        
        print(f"Initialized zarr store at: {zarr_path}")
    
    def get_next_run_index(self) -> int:
        """Get the next available run index in a thread-safe manner."""
        with self._counter_lock:
            run_index = self._run_counter
            self._run_counter += 1
            return run_index
    
    def write(
        self, run_name: str, image: np.ndarray, masks: np.ndarray, 
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
        
        # Get thread-safe run index
        run_index = self.get_next_run_index()
        
        try:
            # Create group for this run - zarr handles concurrent group creation
            # run_group = self.zroot.create_group(f"run_{run_index:06d}")
            run_group = self.zroot.create_group(run_name)   
            
            # Store run metadata
            # run_group.attrs['run_name'] = run_name
            # run_group.attrs['run_index'] = run_index
            if metadata:
                for key, value in metadata.items():
                    run_group.attrs[key] = value
            
            # Write datasets - zarr handles concurrent writes to different groups
            run_group.create_dataset(
                "image", 
                data=image, 
                dtype=image.dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
            )
            
            run_group.create_dataset(
                "masks", 
                data=masks, 
                dtype=masks.dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
            )
            
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
