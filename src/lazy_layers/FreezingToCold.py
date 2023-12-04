import numpy as np
from lazy_layers.layer import Layer

class FreezingToCold(Layer):
    """
    Processing layer in world generation.
    Any freezing land adjacent to a warm or temperate region will turn cold.
    """
    
    def __init__(self):
        """Constructs a new FreezingToCold Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError