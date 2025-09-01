"""Live2D v3 model renderer wrapper."""

from typing import Optional, Tuple, Dict, Union, Iterator, List
import torch
import numpy as np
import pygame
import live2d.v3 as live2d

from ..core.types import Live2DParameters


class Live2DRenderer:
    """
    Wrapper for Live2D v3 model rendering.
    """
    
    def __init__(self, 
                 model_path: str,
                 canvas_size: Tuple[int, int] = (512, 512),
                 background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Initialize Live2D renderer.
        
        Args:
            model_path: Path to .model3.json file
            canvas_size: Rendering canvas size (width, height)
            background_color: RGBA background color
        """
        self.model_path = model_path
        self.canvas_size = canvas_size
        self.background_color = background_color
        
        self.model: Optional[live2d.LAppModel] = None
        self.initialized = False
        
        # Parameter cache for optimization
        self._param_cache: Dict[str, float] = {}
    
    def initialize(self):
        """Initialize pygame and Live2D framework."""
        if self.initialized:
            return
        
        # Initialize pygame
        pygame.init()
        
        # Create OpenGL context
        pygame.display.set_mode(
            self.canvas_size,
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN
        )
        
        # Initialize Live2D
        live2d.init()
        live2d.glInit()
        
        # Load model
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(self.model_path)
        self.model.Resize(*self.canvas_size)
        
        # Disable auto features (we control everything)
        self.model.SetAutoBlinkEnable(False)
        self.model.SetAutoBreathEnable(False)
        
        self.initialized = True
    
    def render(self, input_data: Union[Live2DParameters, Iterator[Optional[Live2DParameters]], List[Optional[Live2DParameters]]]) -> Union[torch.Tensor, Iterator[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Unified rendering interface supporting single parameters, batch, and streaming modes.
        
        Args:
            input_data: Input parameters - single object, list, or iterator of Optional[Live2DParameters]
            
        Returns:
            - Single input: torch.Tensor
            - Multiple inputs: Iterator or List of Optional[torch.Tensor]
        """
        from ..processors.stream_utils import is_iterator, apply_to_stream
        
        # Ensure renderer is initialized
        if not self.initialized:
            self.initialize()
        
        # Handle single parameters
        if isinstance(input_data, Live2DParameters):
            return self._render_single(input_data)
        
        # Handle iterator/generator (streaming mode)
        elif is_iterator(input_data):
            return apply_to_stream(input_data, lambda params: self._render_single(params) if params is not None else None, preserve_none=True)
        
        # Handle list (batch mode) 
        else:
            return [self._render_single(params) if params is not None else None for params in input_data]
    
    def _render_single(self, parameters: Live2DParameters) -> torch.Tensor:
        """
        Render Live2D model with given parameters for a single frame.
        
        Args:
            parameters: Live2D parameters to apply
            
        Returns:
            Rendered image as torch tensor (H, W, C) in RGB format
        """
        # Apply parameters to model
        self._apply_parameters(parameters)
        
        # Clear buffer
        r, g, b, a = [c / 255.0 for c in self.background_color]
        live2d.clearBuffer(r, g, b, a)
        
        # Update and draw model
        self.model.Update()
        self.model.Draw()
        
        # Read pixels from OpenGL buffer
        pixels = self._read_pixels()
        
        return pixels
    
    def render_to_surface(self, parameters: Live2DParameters) -> pygame.Surface:
        """
        Render Live2D model to pygame surface.
        
        Args:
            parameters: Live2D parameters to apply
            
        Returns:
            Rendered pygame surface
        """
        pixels = self._render_single(parameters)
        
        # Convert torch tensor to numpy for pygame compatibility
        pixels_np = pixels.cpu().numpy()
        
        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(pixels_np.swapaxes(0, 1))
        
        return surface
    
    def _apply_parameters(self, parameters: Live2DParameters):
        """Apply parameters to Live2D model."""
        param_dict = parameters.to_dict()
        
        for param_name, value in param_dict.items():
            # Only update if value changed (optimization)
            if param_name in self._param_cache and self._param_cache[param_name] == value:
                continue
            
            # Apply parameter to model
            try:
                self.model.SetParameterValue(param_name, value, 1.0)
                self._param_cache[param_name] = value
            except Exception as e:
                # Parameter might not exist in this model
                pass
    
    def _read_pixels(self) -> torch.Tensor:
        """Read pixels from OpenGL framebuffer."""
        import OpenGL.GL as gl
        
        width, height = self.canvas_size
        
        # Read pixels from framebuffer
        pixels = gl.glReadPixels(
            0, 0, width, height,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        
        # Convert to numpy first to avoid buffer warning, then to CUDA tensor
        pixels_np = np.frombuffer(pixels, dtype=np.uint8).copy()
        pixels_tensor = torch.from_numpy(pixels_np).cuda()
        pixels_tensor = pixels_tensor.reshape((height, width, 3))
        
        # Flip vertically (OpenGL convention) - GPU accelerated
        pixels_tensor = torch.flip(pixels_tensor, dims=[0])
        
        # Convert RGB to BGR using tensor operations - GPU accelerated  
        pixels_bgr = pixels_tensor[:, :, [2, 1, 0]]  # Swap R and B channels
        
        return pixels_bgr
    
    def set_expression(self, expression_name: str):
        """
        Set model expression.
        
        Args:
            expression_name: Name of expression to apply
        """
        if self.model:
            self.model.SetExpression(expression_name)
    
    def set_random_expression(self):
        """Set a random expression."""
        if self.model:
            self.model.SetRandomExpression()
    
    def start_motion(self, group: str = "Idle", priority: int = 2):
        """
        Start a motion.
        
        Args:
            group: Motion group name
            priority: Motion priority
        """
        if self.model:
            self.model.StartMotion(group, priority)
    
    def start_random_motion(self, group: str = "Idle", priority: int = 2):
        """
        Start a random motion from group.
        
        Args:
            group: Motion group name
            priority: Motion priority
        """
        if self.model:
            self.model.StartRandomMotion(group, priority)
    
    def cleanup(self):
        """Clean up resources."""
        if self.initialized:
            live2d.dispose()
            pygame.quit()
            self.initialized = False
            self.model = None
            self._param_cache.clear()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()