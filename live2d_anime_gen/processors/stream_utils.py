"""Streaming utilities for unified batch/stream processing."""

from typing import Any, Iterator, Callable, TypeVar

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