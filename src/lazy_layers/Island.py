import numpy as np
from numpy.random import Generator
from lazy_layers.layer import Layer
from core import cell

class Island(Layer):
    """
    Processing layer in world generation for adding an initial
    set of islands to the map.
    """
    INIT_DIM = 4
    
    def __init__(self):
        """Constructs a new Island Layer Object"""
    
    def run(self, dims: int, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        Implemented in child classes
        :param dims: the resulting dimensions of the board
        :param rng: a random number generator to use
        :return: A new copy of the board after the transformation
        """
        PROB_LAND = 0.1
        
        board = np.zeros((dims, dims), dtype=cell)
        for i in range(dims):
            for j in range(dims):
                board[i][j] = cell.LAND if rng.uniform(0.0, 1.0) <= PROB_LAND else cell.OCEAN

        return board