import numpy as np
form core import is_edge_cell
from lazy_layers.layer import Layer

class Shore(Layer):
    """
    Processing layer in world generation.
    Add shore on edge of land and ocean.
    """
    DEPTH = 3
    SWAMP_DEPTH = 5  # Increased depth for swamp shores, adjust as needed

    
    def __init__(self):
        """Constructs a new Shore Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
	# Iterate over each cell in the board
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                # Check if the cell is at the edge
                if is_edge_cell(board, x, y):
                    # Determine if the cell is a swamp shore or a regular shore
                    if board[x,y] == Cell.SWAMP:
                        board[x, y] = Cell.SWAMP_SHORE
                    else:
                        board[x, y] = Cell.SHORE
        return board