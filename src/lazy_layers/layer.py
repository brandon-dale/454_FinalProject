import numpy as np

class Layer:
    """
    Interface class for a processing layer in world generation
    """
    def __init__(self):
        """Constructs a new Layer Object"""
    
    def run(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        Implemented in child classes
        """
        raise NotImplementedError
