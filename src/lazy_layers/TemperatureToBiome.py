import numpy as np
from layer import Layer

class TemperatureToBiome(Layer):
    """
    Processing layer in world generation.
    """
    
    def __init__(self):
        """Constructs a new TemperatureToBiome Layer Object"""
    
    def run(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError