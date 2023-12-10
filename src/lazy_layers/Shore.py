import numpy as np
from core import is_edge_cell, Cell, set_board_region
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
        n_rows, n_cols = board.shape
        EXPAND_RAD = 1
        
        def is_edge_cell(r, c):
            if board[r][c] != Cell.OCEAN:
                # Check adjacent cells for ocean
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and board[nr][nc] == Cell.OCEAN:
                        return True
            return False
        
	    # Iterate over each cell in the board
        ignore_cells = set([Cell.TUNDRA, Cell.ICE_PLAINS, Cell.TAIGA, Cell.SNOWY_FOREST])
     
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                # Check if the cell is at the edge
                if is_edge_cell(x, y):
                    # Determine if the cell is a swamp shore or a regular shore
                    if board[x,y] == Cell.SWAMP:
                        board = set_board_region(board, x, y, EXPAND_RAD, Cell.SWAMP_SHORE)
                    elif board[x,y] in ignore_cells:
                        continue
                    else:
                        board = set_board_region(board, x, y, EXPAND_RAD, Cell.SHORE)
        return board