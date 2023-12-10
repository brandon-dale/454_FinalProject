import numpy as np
from lazy_layers.layer import Layer
import copy
from core import Cell, get_neighbors

class RemoveTooMuchOcean(Layer):
    """
    Processing layer in world generation.
    All ocean regions surrounded by more ocean have a P_LAND chance of becoming land.
    Only check up/down and left/right.
    """
    P_LAND = 0.35
    
    def __init__(self):
        """Constructs a new RemoveTooMuchOcean Layer Object"""
        super().__init__()

    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step.
        :param board: A 2D array containing the input board state
        :param rng: A random number generator
        :return: A new copy of the board after the transformation
        """
        new_board = copy.deepcopy(board)
        rows, cols = board.shape
        
        # def is_deep_ocean(x, y):
        #     SEARCH_RADIUS = 2
        #     neighbors = get_neighbors(board, x, y, SEARCH_RADIUS).flatten()
        #     return np.unique(neighbors) == 1
            
        for i in range(rows):
            for j in range(cols):
                if board[i, j] == Cell.OCEAN:  # Assuming 0 represents ocean
                    # Check if surrounded by ocean
                    if self.is_surrounded_by_ocean(board, i, j):
                        # 50% chance to become land
                        if rng.uniform(0., 1.) < RemoveTooMuchOcean.P_LAND:
                            new_board[i, j] = Cell.LAND  # Assuming 1 represents land

        return new_board

    def is_surrounded_by_ocean(self, board, i, j):
        """
        Check if the cell is surrounded by ocean in up/down and left/right.
        """
        rows, cols = board.shape
        # Check up, down, left, and right cells
        if i > 0 and board[i-1, j] != Cell.OCEAN:  # Check up
            return False
        if i < rows - 1 and board[i+1, j] != Cell.OCEAN:  # Check down
            return False
        if j > 0 and board[i, j-1] != Cell.OCEAN:  # Check left
            return False
        if j < cols - 1 and board[i, j+1] != Cell.OCEAN:  # Check right
            return False
        return True
