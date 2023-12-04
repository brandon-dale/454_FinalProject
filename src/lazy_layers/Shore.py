import numpy as np
from layer import Layer

class Shore(Layer):
    """
    Processing layer in world generation.
    Add shore on edge of land and ocean.
    """
    DEPTH = 3
    
    def __init__(self):
        """Constructs a new Shore Layer Object"""
    
    def run(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError