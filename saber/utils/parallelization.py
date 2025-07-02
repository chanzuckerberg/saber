"""
parallelization.py - Flexible GPU Processing
==========================================

Choose between threading (fast) or multiprocessing (robust) approaches.
Perfect for any function, including your segment_tomogram_separate_process.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional, Union
import torch, threading, time, queue, os
import torch.multiprocessing as mp
from tqdm import tqdm

class GPUPool:
    """
    Flexible GPU processing pool supporting both threading and multiprocessing.
    
    Usage:
        # Fast threading approach (shared models)
        pool = GPUPool(n_gpus=4, approach="threading")
        
        # Robust multiprocessing approach (isolated models)  
        pool = GPUPool(n_gpus=4, approach="multiprocessing")
        
        # Auto-choose based on model size
        pool = GPUPool(n_gpus=4, approach="auto", model_size_gb=20)
    """
    
    def __init__(self, 
                 approach: str = "threading",  # "threading", "multiprocessing", or "auto"
                 init_fn: Optional[Callable] = None,
                 init_args: tuple = (),
                 init_kwargs: dict = {},
                 verbose: bool = True):
        
        self.n_gpus = torch.cuda.device_count()
        self.init_fn = init_fn
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.verbose = verbose
        
        self.approach = approach
        
        if approach == "threading":
            self._init_threading()
        elif approach == "multiprocessing":
            self._init_multiprocessing()
        else:
            raise ValueError("approach must be 'threading', 'multiprocessing', or 'auto'")
    
    # ============================================================================
    # THREADING IMPLEMENTATION
    # ============================================================================
    
    def _init_threading(self):
        """Initialize threading approach - shared models"""
        self.models = {}  # Shared across threads
        self.model_locks = {}  # Per-GPU locks
        self.initialized = threading.Event()
        
        if self.verbose:
            print(f"GPUPool (threading): {self.n_gpus} GPUs with shared models")
    
    def _initialize_models_threading(self):
        """Load models once, shared across all threads"""
        if self.verbose:
            print("Loading models for threading approach...")
            
        for gpu_id in range(self.n_gpus):
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            if self.init_fn:
                if self.verbose:
                    print(f"GPU {gpu_id}: Loading shared models...")
                start_time = time.time()
                models = self.init_fn(gpu_id, *self.init_args, **self.init_kwargs)
                load_time = time.time() - start_time
                
                self.models[gpu_id] = models
                self.model_locks[gpu_id] = threading.RLock()
                
                if self.verbose:
                    gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1e9
                    print(f"GPU {gpu_id}: Models loaded in {load_time:.1f}s, {gpu_mem:.1f}GB VRAM")
            else:
                self.models[gpu_id] = None
                self.model_locks[gpu_id] = threading.RLock()
                
        self.initialized.set()
        if self.verbose:
            print("All shared models ready!")
    
    def _execute_threading(self, func, tasks, task_ids, progress_desc):
        """Execute using threading approach"""
        if not self.initialized.is_set():
            self._initialize_models_threading()
            
        def worker_thread(task_data):
            task_id, gpu_id, args, kwargs = task_data
            
            # Wait for initialization
            self.initialized.wait()
            
            # Get exclusive access to this GPU
            with self.model_locks[gpu_id]:
                try:
                    torch.cuda.set_device(gpu_id)
                    
                    # Add GPU context
                    enhanced_kwargs = kwargs.copy()
                    enhanced_kwargs['gpu_id'] = gpu_id
                    if self.models[gpu_id] is not None:
                        enhanced_kwargs['models'] = self.models[gpu_id]
                    
                    start_time = time.time()
                    result = func(*args, **enhanced_kwargs)
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'processing_time': processing_time,
                        'result': result
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'error': str(e)
                    }
        
        # Prepare tasks with GPU assignment
        prepared_tasks = []
        for i, (task_id, task) in enumerate(zip(task_ids, tasks)):
            gpu_id = i % self.n_gpus  # Round-robin
            
            if isinstance(task, dict):
                args, kwargs = (), task
            elif isinstance(task, tuple) and len(task) == 2 and isinstance(task[1], dict):
                args, kwargs = task
            elif isinstance(task, (list, tuple)):
                args, kwargs = task, {}
            else:
                args, kwargs = (task,), {}
                
            prepared_tasks.append((task_id, gpu_id, args, kwargs))
        
        # Execute with thread pool
        results = []
        with ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            with tqdm(total=len(tasks), desc=progress_desc, unit='task', disable=not self.verbose) as pbar:
                future_to_task = {
                    executor.submit(worker_thread, task): task 
                    for task in prepared_tasks
                }
                
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    
                    if result['success'] and self.verbose:
                        pbar.set_postfix({
                            'GPU': result['gpu_id'],
                            'Task': str(result['task_id'])[:15],
                            'Time': f"{result['processing_time']:.1f}s"
                        })
                    elif not result['success'] and self.verbose:
                        print(f"\n❌ Task {result['task_id']} failed: {result['error']}")
                        
                    if self.verbose:
                        pbar.update(1)
        
        return results
    
    # ============================================================================
    # MULTIPROCESSING IMPLEMENTATION
    # ============================================================================
    
    def _init_multiprocessing(self):
        """Initialize multiprocessing approach - isolated models"""
        self.task_queue = mp.Queue(maxsize=self.n_gpus * 2)
        self.result_queue = mp.Queue()
        self.workers = []
        self.shutdown_event = mp.Event()
        self.started = False
        
        if self.verbose:
            print(f"GPUPool (multiprocessing): {self.n_gpus} GPUs with isolated models")
    
    def _worker_process(self, gpu_id: int):
        """Multiprocessing worker - each has its own models"""
        try:
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            # Load models once per process
            models = None
            if self.init_fn:
                if self.verbose:
                    print(f"GPU {gpu_id}: Loading isolated models...")
                start_time = time.time()
                models = self.init_fn(gpu_id, *self.init_args, **self.init_kwargs)
                load_time = time.time() - start_time
                
                if self.verbose:
                    gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1e9
                    print(f"GPU {gpu_id}: Models loaded in {load_time:.1f}s, {gpu_mem:.1f}GB VRAM")
            
            if self.verbose and not self.init_fn:
                print(f"GPU {gpu_id}: Worker ready (no models)")
            
            # Process tasks
            task_count = 0
            while not self.shutdown_event.is_set():
                try:
                    task_data = self.task_queue.get(timeout=1.0)
                    
                    if task_data is None:  # Shutdown signal
                        break
                    
                    task_id, func, args, kwargs = task_data
                    task_count += 1
                    
                    # Execute task
                    start_time = time.time()
                    torch.cuda.set_device(gpu_id)
                    
                    try:
                        # Add GPU context
                        enhanced_kwargs = kwargs.copy()
                        enhanced_kwargs['gpu_id'] = gpu_id
                        if models is not None:
                            enhanced_kwargs['models'] = models
                        
                        result = func(*args, **enhanced_kwargs)
                        processing_time = time.time() - start_time
                        
                        self.result_queue.put({
                            'success': True,
                            'task_id': task_id,
                            'gpu_id': gpu_id,
                            'processing_time': processing_time,
                            'result': result
                        })
                        
                    except Exception as e:
                        self.result_queue.put({
                            'success': False,
                            'task_id': task_id,
                            'gpu_id': gpu_id,
                            'error': str(e)
                        })
                    
                    # Periodic cleanup
                    if task_count % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.verbose:
                        print(f"GPU {gpu_id} worker error: {e}")
                    break
            
        except Exception as e:
            self.result_queue.put({
                'success': False,
                'task_id': 'INIT_FAILED',
                'gpu_id': gpu_id,
                'error': f"Worker initialization failed: {str(e)}"
            })
        
        finally:
            if self.verbose:
                print(f"GPU {gpu_id}: Worker shutting down (processed {task_count} tasks)")
            try:
                torch.cuda.empty_cache()
            except:
                pass
    
    def _execute_multiprocessing(self, func, tasks, task_ids, progress_desc):
        """Execute using multiprocessing approach"""
        if not self.started:
            self.start()
        
        # Submit tasks
        for task_id, task in zip(task_ids, tasks):
            if isinstance(task, dict):
                args, kwargs = (), task
            elif isinstance(task, tuple) and len(task) == 2 and isinstance(task[1], dict):
                args, kwargs = task
            elif isinstance(task, (list, tuple)):
                args, kwargs = task, {}
            else:
                args, kwargs = (task,), {}
            
            self.task_queue.put((task_id, func, args, kwargs))
        
        # Collect results
        results = []
        failed_count = 0
        
        with tqdm(total=len(tasks), desc=progress_desc, unit='task', disable=not self.verbose) as pbar:
            for _ in range(len(tasks)):
                result = self.result_queue.get()
                results.append(result)
                
                if result['success']:
                    if self.verbose:
                        pbar.set_postfix({
                            'GPU': result['gpu_id'],
                            'Task': str(result['task_id'])[:15],
                            'Time': f"{result['processing_time']:.1f}s",
                            'Failed': failed_count
                        })
                else:
                    failed_count += 1
                    if result['task_id'] == 'INIT_FAILED':
                        print(f"\n❌ GPU {result['gpu_id']} initialization failed: {result['error']}")
                    elif self.verbose:
                        print(f"\n❌ Task {result['task_id']} failed: {result['error']}")
                
                if self.verbose:
                    pbar.update(1)
        
        return results
    
    def start(self):
        """Start workers (only needed for multiprocessing)"""
        if self.approach == "multiprocessing" and not self.started:
            if self.verbose:
                print(f"Starting {self.n_gpus} multiprocessing workers...")
            
            mp.set_start_method('spawn', force=True)
            for gpu_id in range(self.n_gpus):
                worker = mp.Process(target=self._worker_process, args=(gpu_id,))
                worker.start()
                self.workers.append(worker)
            
            self.started = True
            time.sleep(0.5)  # Let workers initialize
    
    # ============================================================================
    # UNIFIED INTERFACE
    # ============================================================================
    
    def execute(self, 
                func: Callable,
                tasks: List[Any],
                task_ids: Optional[List] = None,
                progress_desc: str = "Processing") -> List[Dict]:
        """
        Execute function on all tasks across GPUs.
        
        Your function will receive:
            - All your original arguments
            - gpu_id: int (keyword argument)
            - models: Any (keyword argument, if init_fn was provided)
        """
        if not tasks:
            return []
        
        if task_ids is None:
            task_ids = list(range(len(tasks)))
        
        if self.approach == "threading":
            results = self._execute_threading(func, tasks, task_ids, progress_desc)
        else:
            results = self._execute_multiprocessing(func, tasks, task_ids, progress_desc)
        
        # Print statistics
        if self.verbose:
            self._print_stats(results)
        
        return sorted(results, key=lambda x: x.get('task_id', 0))
    
    def _print_stats(self, results):
        """Print execution statistics"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{'='*50}")
        print(f"EXECUTION COMPLETE ({self.approach.upper()})")
        print(f"{'='*50}")
        print(f"Total tasks: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print(f"Failed runs: {[r['task_id'] for r in failed]}")
            for failed_run in failed:
                print(f"  - {failed_run['task_id']}: {failed_run['error']}")        
        
        if successful:
            gpu_stats = {}
            for r in successful:
                gpu_id = r['gpu_id']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {'count': 0, 'total_time': 0.0}
                gpu_stats[gpu_id]['count'] += 1
                gpu_stats[gpu_id]['total_time'] += r['processing_time']
            
            print(f"\nGPU Statistics:")
            for gpu_id, stats in gpu_stats.items():
                avg_time = stats['total_time'] / stats['count']
                print(f"  GPU {gpu_id}: {stats['count']} tasks, avg {avg_time:.2f}s/task")
    
    def shutdown(self):
        """Shutdown workers"""
        if self.approach == "multiprocessing" and self.started:
            if self.verbose:
                print("Shutting down multiprocessing workers...")
            
            self.shutdown_event.set()
            
            # Send shutdown signals
            for _ in range(self.n_gpus):
                try:
                    self.task_queue.put(None, timeout=1.0)
                except:
                    pass
            
            # Wait for workers
            for worker in self.workers:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            
            self.started = False
            if self.verbose:
                print("All workers shut down")
    
    def __enter__(self):
        if self.approach == "multiprocessing":
            self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def gpu_map(func: Callable, 
            tasks: List[Any],
            approach: str = "auto",
            model_size_gb: float = 1.0,
            n_gpus: Optional[int] = None,
            init_fn: Optional[Callable] = None,
            init_args: tuple = (),
            init_kwargs: dict = {},
            verbose: bool = True) -> List[Dict]:
    """
    Convenience function for GPU mapping with automatic approach selection.
    
    Example:
        # Let it auto-choose based on model size
        results = gpu_map(my_function, tasks, model_size_gb=20)  # Uses multiprocessing
        
        # Force threading for speed
        results = gpu_map(my_function, tasks, approach="threading")
    """
    with GPUPool(n_gpus, approach, init_fn, init_args, init_kwargs, model_size_gb, verbose) as pool:
        return pool.execute(func, tasks)

if __name__ == "__main__":
    print("Flexible GPU Processing Pool")
    print("=" * 40)
    print("Supports both threading (fast) and multiprocessing (robust) approaches")
    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA GPUs")
    else:
        print("No CUDA GPUs available")