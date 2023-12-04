import numpy as np
from layer import Layer

class RemoveTooMuchOcean(Layer):
    """
    Processing layer in world generation.
    All ocean regions surrounded by more ocean have a 50% chance of becoming land.
    Only check up/down and left/right.
    """
    
    def __init__(self):
        """Constructs a new RemoveTooMuchOcean Layer Object"""
    
    def run(board: np.ndarray) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError