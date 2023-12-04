import numpy as np
import core as cell
from layer import Layer

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
        rows, cols = board.shape

        # Iterating over the edge cells
        for i in range(rows):
            for j in [0, cols-1]:  # First and last column
                board[i][j] = cell.LAND if rng.random() < 0.75 else cell.OCEAN
        return board