import numpy as np
from core import Cell
from lazy_layers.layer import Layer
import copy

class AddIsland(Layer):
    """
    Processing layer in world generation.
    Each edge cell between land and ocean has a 75% chance of becoming land and a 25% chance of becoming ocean.
    """
    
    def __init__(self):
        """Constructs a new AddIsland Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :param rng: A random number generator
        :return: A new copy of the board after the transformation
        """
        next_board = copy.deepcopy(board)
        rows, cols = board.shape

        # Function to determine if a cell is an edge between land and ocean
        def is_edge_cell(r, c):
            if board[r][c] == Cell.LAND:
                # Check adjacent cells for ocean
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == Cell.OCEAN:
                        return True
            elif board[r][c] == Cell.OCEAN:
                # Check adjacent cells for land
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == Cell.LAND:
                        return True
            return False

        # Iterating over the edge cells
        for i in range(rows):
            for j in range(cols):
                if is_edge_cell(i, j):
                    next_board[i][j] = Cell.LAND if rng.uniform(0.0, 1.0) < 0.75 else Cell.OCEAN

        return next_board