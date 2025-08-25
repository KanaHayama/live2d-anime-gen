"""Parameter smoothing utilities for temporal filtering."""

from typing import Optional, Deque
from collections import deque
import torch

from ..core.types import Live2DParameters


class ParameterSmoother:
    """
    Advanced parameter smoothing with multiple filtering options.
    """
    
    def __init__(self, 
                 method: str = "ema",
                 window_size: int = 5,
                 alpha: float = 0.5):
        """
        Initialize parameter smoother.
        
        Args:
            method: Smoothing method ("ema", "sma", "kalman")
            window_size: Window size for moving average methods
            alpha: Smoothing factor for EMA (0-1)
        """
        self.method = method
        self.window_size = window_size
        self.alpha = alpha
        
        # State for different methods
        self.history: Deque[Live2DParameters] = deque(maxlen=window_size)
        self.prev_params: Optional[Live2DParameters] = None
        
        # Kalman filter states (simplified)
        self.kalman_state: Optional[Live2DParameters] = None
        self.kalman_covariance: float = 1.0
    
    def smooth(self, params: Live2DParameters) -> Live2DParameters:
        """
        Apply smoothing to parameters.
        
        Args:
            params: Current frame parameters
            
        Returns:
            Smoothed parameters
        """
        if self.method == "ema":
            return self._exponential_moving_average(params)
        elif self.method == "sma":
            return self._simple_moving_average(params)
        elif self.method == "kalman":
            return self._kalman_filter(params)
        else:
            return params
    
    def _exponential_moving_average(self, params: Live2DParameters) -> Live2DParameters:
        """Exponential moving average smoothing."""
        if self.prev_params is None:
            self.prev_params = params.clone()
            return params
        
        smoothed = params.clone()
        alpha = self.alpha
        
        # Apply EMA to each parameter
        for field in params.__dataclass_fields__:
            if field == "custom_params":
                continue
                
            curr_val = getattr(params, field)
            prev_val = getattr(self.prev_params, field)
            
            if curr_val is not None and prev_val is not None:
                smoothed_val = alpha * prev_val + (1 - alpha) * curr_val
                setattr(smoothed, field, smoothed_val)
        
        self.prev_params = smoothed.clone()
        return smoothed
    
    def _simple_moving_average(self, params: Live2DParameters) -> Live2DParameters:
        """Simple moving average smoothing."""
        self.history.append(params.clone())
        
        if len(self.history) < 2:
            return params
        
        smoothed = params.clone()
        
        # Average over history window
        for field in params.__dataclass_fields__:
            if field == "custom_params":
                continue
            
            values = []
            for hist_params in self.history:
                val = getattr(hist_params, field)
                if val is not None:
                    values.append(val)
            
            if values:
                avg_val = torch.stack(values).mean(dim=0)
                setattr(smoothed, field, avg_val)
        
        return smoothed
    
    def _kalman_filter(self, params: Live2DParameters) -> Live2DParameters:
        """Simplified Kalman filter for parameter smoothing."""
        if self.kalman_state is None:
            self.kalman_state = params.clone()
            return params
        
        # Simplified 1D Kalman filter for each parameter
        process_noise = 0.01
        measurement_noise = 0.1
        
        smoothed = params.clone()
        
        for field in params.__dataclass_fields__:
            if field == "custom_params":
                continue
            
            measurement = getattr(params, field)
            prediction = getattr(self.kalman_state, field)
            
            if measurement is not None and prediction is not None:
                # Predict
                pred_covariance = self.kalman_covariance + process_noise
                
                # Update
                gain = pred_covariance / (pred_covariance + measurement_noise)
                estimate = prediction + gain * (measurement - prediction)
                self.kalman_covariance = (1 - gain) * pred_covariance
                
                setattr(smoothed, field, estimate)
                setattr(self.kalman_state, field, estimate)
        
        return smoothed
    
    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.prev_params = None
        self.kalman_state = None
        self.kalman_covariance = 1.0