"""Streaming utilities for unified batch/stream processing."""

from typing import Any, Iterator, Union, List, Optional, Callable, TypeVar

T = TypeVar('T')


def is_iterator(obj: Any) -> bool:
    """
    Check if object is an iterator (excluding string, bytes, dict).
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is an iterator that should be treated as stream
    """
    # Exclude common non-stream iterables
    if isinstance(obj, (str, bytes, dict, list, tuple, set)):
        return False
    
    # Check if it has __iter__ method (is iterable)
    return hasattr(obj, '__iter__')


def ensure_iterator(data: Union[T, List[T], Iterator[T]]) -> Iterator[T]:
    """
    Convert input to iterator for unified processing.
    
    Args:
        data: Input data (single item, list, or iterator)
        
    Returns:
        Iterator over the data
    """
    if is_iterator(data):
        return data  # Already an iterator
    elif isinstance(data, (list, tuple)):
        return iter(data)
    else:
        # Single item
        return iter([data])


def collect_stream(stream: Iterator[T], buffer_size: Optional[int] = None) -> Union[List[T], Iterator[T]]:
    """
    Collect stream data based on buffer size.
    
    Args:
        stream: Input stream
        buffer_size: None for full collection, number for batch size, 1 for passthrough
        
    Returns:
        List (if buffer_size is None) or Iterator
    """
    if buffer_size is None:
        # Full collection for batch processing
        return list(stream)
    elif buffer_size == 1:
        # Pure streaming - passthrough
        return stream
    else:
        # Batched streaming
        return batch_stream(stream, buffer_size)


def batch_stream(stream: Iterator[T], batch_size: int) -> Iterator[List[T]]:
    """
    Convert stream to batched stream.
    
    Args:
        stream: Input stream
        batch_size: Size of each batch
        
    Yields:
        Batches of items from stream
    """
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield remaining items if any
    if batch:
        yield batch


def apply_to_stream(stream: Iterator[T], 
                   func: Callable[[T], Any],
                   preserve_none: bool = True) -> Iterator[Any]:
    """
    Apply function to each item in stream.
    
    Args:
        stream: Input stream
        func: Function to apply to each item
        preserve_none: If True, None values pass through unchanged
        
    Yields:
        Results of applying func to each stream item
    """
    for item in stream:
        if item is None and preserve_none:
            yield None
        else:
            yield func(item)


def filter_stream(stream: Iterator[T], 
                 predicate: Callable[[T], bool]) -> Iterator[T]:
    """
    Filter stream items based on predicate.
    
    Args:
        stream: Input stream
        predicate: Function to test each item
        
    Yields:
        Items that pass the predicate test
    """
    for item in stream:
        if predicate(item):
            yield item


def enumerate_stream(stream: Iterator[T], 
                    start: int = 0) -> Iterator[tuple[int, T]]:
    """
    Add indices to stream items.
    
    Args:
        stream: Input stream
        start: Starting index
        
    Yields:
        Tuples of (index, item)
    """
    for i, item in enumerate(stream, start):
        yield i, item