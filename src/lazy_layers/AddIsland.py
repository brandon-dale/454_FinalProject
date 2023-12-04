import numpy as np
from layer import Layer

class AddIsland(Layer):
    """
    Processing layer in world generation.    
    Each edge cell has a random probability of toggling.
    """
    
    def __init__(self):
        """Constructs a new AddIsland Layer Object"""
    
    def run(board: np.ndarray) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError