"""Input type enumeration for pipeline processing."""

from enum import Enum


class InputType(Enum):
    """Enumeration of supported input types for the pipeline."""
    VIDEO = "video"
    LANDMARKS = "landmarks"  
    PARAMETERS = "parameters"