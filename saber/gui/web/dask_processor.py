# === annotation_gui/dask_processor.py ===

"""Dask integration for distributed processing."""

import logging
from dask.distributed import Client, as_completed
from dask import delayed
import numpy as np
import zarr
import json

logger = logging.getLogger(__name__)

class DaskProcessor:
    """Handle Dask-based distributed processing for annotations."""
    
    def __init__(self, scheduler=None, n_workers=4):
        """
        Initialize Dask processor.
        
        Args:
            scheduler: Dask scheduler address or None for local cluster
            n_workers: Number of workers for local cluster
        """
        self.scheduler = scheduler
        self.n_workers = n_workers
        self.client = None
    
    def start(self):
        """Start or connect to Dask cluster."""
        try:
            if self.scheduler:
                logger.info(f"Connecting to Dask scheduler: {self.scheduler}")
                self.client = Client(self.scheduler)
            else:
                logger.info(f"Starting local Dask cluster with {self.n_workers} workers")
                self.client = Client(n_workers=self.n_workers, threads_per_worker=2)
            
            logger.info(f"Dask dashboard: {self.client.dashboard_link}")
            
        except Exception as e:
            logger.error(f"Failed to start Dask: {e}")
            self.client = None
    
    def close(self):
        """Close Dask client and cluster."""
        if self.client:
            self.client.close()
    
    def get_status(self):
        """Get Dask cluster status."""
        if not self.client:
            return {'status': 'not connected'}
        
        try:
            info = self.client.scheduler_info()
            return {
                'status': 'connected',
                'workers': len(info['workers']),
                'dashboard': self.client.dashboard_link,
                'scheduler': str(self.client.scheduler)
            }
        except:
            return {'status': 'error'}
    
    @delayed
    def process_mask(self, mask_data, threshold=0.5):
        """Process a single mask (Dask delayed function)."""
        # Example processing - you can add more complex operations
        processed = mask_data > threshold
        return processed.astype(np.uint8)
    
    def load_run_data(self, zarr_root, run_id):
        """Load and process run data using Dask."""
        if not self.client:
            raise RuntimeError("Dask client not initialized")
        
        # Create delayed tasks
        image_task = delayed(zarr_root[run_id]['image'][:])
        masks_task = delayed(zarr_root[run_id].get('labels', 
                            zarr_root[run_id].get('masks', []))[:])
        
        # Submit to cluster
        future_image = self.client.compute(image_task)
        future_masks = self.client.compute(masks_task)
        
        # Get results
        image = future_image.result()
        masks = future_masks.result()
        
        # Process masks in parallel
        if len(masks) > 0:
            mask_tasks = [self.process_mask(mask) for mask in masks]
            futures = self.client.compute(mask_tasks)
            processed_masks = self.client.gather(futures)
            masks = np.array(processed_masks)
        
        # Transpose if needed
        if image.shape[0] < image.shape[1]:
            image = image.T
            masks = np.swapaxes(masks, 1, 2)
        
        return {
            'image': image.tolist(),
            'masks': masks.tolist(),
            'shape': image.shape
        }
    
    def save_annotation(self, output_path, run_id, class_dict, accepted_masks):
        """Save annotation using Dask for parallel processing."""
        if not self.client:
            raise RuntimeError("Dask client not initialized")
        
        @delayed
        def save_to_zarr(path, run_id, data):
            output_zarr = zarr.open(str(path), mode='a')
            group = output_zarr.require_group(run_id)
            group.attrs['class_dict'] = json.dumps(data['class_dict'])
            group.attrs['accepted_masks'] = data['accepted_masks']
            return {'status': 'saved', 'run_id': run_id}
        
        # Submit save task
        task = save_to_zarr(output_path, run_id, {
            'class_dict': class_dict,
            'accepted_masks': accepted_masks
        })
        
        future = self.client.compute(task)
        return future.result()
    
    def batch_process_runs(self, zarr_root, run_ids, process_func):
        """Process multiple runs in parallel using Dask."""
        if not self.client:
            raise RuntimeError("Dask client not initialized")
        
        # Create delayed tasks for each run
        tasks = []
        for run_id in run_ids:
            task = delayed(process_func)(zarr_root, run_id)
            tasks.append(task)
        
        # Submit all tasks
        futures = self.client.compute(tasks)
        
        # Process results as they complete
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Processed run: {result.get('run_id')}")
            except Exception as e:
                logger.error(f"Error processing run: {e}")
        
        return results
