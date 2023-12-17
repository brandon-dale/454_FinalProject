import numpy as np
from typing import Any

class Layer:
    """
    Interface class for a processing layer in world generation
    """
    def __init__(self):
        """Constructs a new Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> Any:
        """
        Runs the layer for a single step
        Implemented in child classes
        """
        raise NotImplementedError
