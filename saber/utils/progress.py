"""
Progress bar utility using Rich.

Example:
    from saber.utils.progress import _progress

    for key in _progress(train_keys, description="My Process Description.."):
        ...
"""

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

def _progress(iterable, description="Processing", total=None):
    """
    Wrap an iterable with a Rich progress bar.

    Args:
        iterable: Any iterable object (e.g., list, generator).
        description: Text label to display above the progress bar.
        total: Optional total count. If not provided, will try to use len(iterable).

    Yields:
        Each item from the iterable, while updating the progress bar.

    Example:
        for x in _progress(range(10), "Doing work"):
            time.sleep(0.5)
        
        # With generators/futures that don't have len()
        for future in _progress(as_completed(futures), total=len(futures), description="Processing"):
            result = future.result()
    """

    console = Console()
    
    # Determine the total count
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            # If iterable doesn't have len(), we can't show progress percentage
            total = None

    # The generator itself yields items while advancing the progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)