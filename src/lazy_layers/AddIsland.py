import numpy as np
from core import cell
from lazy_layers.layer import Layer
import copy

class AddIsland(Layer):
    """
    Processing layer in world generation.    
    Each edge cell has a random probability of toggling.
    """
    
    def __init__(self):
        """Constructs a new AddIsland Layer Object"""
    
    def run(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :param rng: A random number generator
        :return: A new copy of the board after the transformation
        """
        next_board = copy.deepcopy(board)
        rows, cols = board.shape

        # Iterating over the edge cells
        for i in range(rows):
            for j in range(cols):
                next_board[i][j] = cell.LAND if rng.uniform(0.0, 1.0) < 0.75 else cell.OCEAN
        return next_board