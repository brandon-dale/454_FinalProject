
import numpy as np
from layer import Layer

class Island(Layer):
    """
    Processing layer in world generation for adding an initial
    set of islands to the map.
    """
    INIT_DIM = 4
    
    def __init__(self):
        """Constructs a new Island Layer Object"""
    
    def run(dims: int=INIT_DIM) -> np.ndarray:
        """
        Runs the layer for a single step
        Implemented in child classes
        :param dims: the resulting dimensions of the board
        :return: A new copy of the board after the transformation
        """
        raise NotImplementedError