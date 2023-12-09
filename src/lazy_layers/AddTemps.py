import numpy as np
from core import Cell
from lazy_layers.layer import Layer
import copy

class AddTemps(Layer):
    """
    Processing layer in world generation.
    Each edge cell between land and ocean has a 75% chance of becoming land and a 25% chance of becoming ocean.
    """
    
    def __init__(self):
        """Constructs a new AddIsland Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError