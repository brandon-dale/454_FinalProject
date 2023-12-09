import numpy as np
from lazy_layers.layer import Layer
from core import Cell, is_edge_cell, set_board_region
import copy


class WarmToTemperate(Layer):
    """
    Processing layer in world generation.
    Any warm land adjacent to a cool or freezing region will turn into a temperate one instead.
    """
    
    def __init__(self):
        """Constructs a new WarmToTemperate Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        next_board = copy.deepcopy(board)
        rows, cols = board.shape
        
        group_a = set([Cell.WARM])
        group_b = set([Cell.COLD, Cell.FREEZING])
        
        for i in range(rows):
            for j in range(cols):
                curr: Cell = board[i][j]
                if is_edge_cell(board, i, j, group_a, group_b):
                    next_board = set_board_region(board, i, j, 1, Cell.TEMPERATE)
        
        return next_board
