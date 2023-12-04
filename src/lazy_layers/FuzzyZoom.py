import numpy as np
from layer import Layer

class FuzzyZoom(Layer):
    """
    Processing layer in world generation.    
    Scale by a factor of 2 and for each edge cell:
        Generate xoff, yoff [-1, 1] uniformly.
        Cell[i][j]<t+t> = Cell[i + xoff][j + yoff]
    """
    
    def __init__(self):
        """Constructs a new FuzzyZoom Layer Object"""
    
    def run(board: np.ndarray) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError