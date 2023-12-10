import numpy as np
from lazy_layers.layer import Layer
from core import Cell, is_edge_cell, set_board_region
import copy

class FreezingToCold(Layer):
    """
    Processing layer in world generation.
    Any freezing land adjacent to a warm or temperate region will turn cold.
    Convert all things within a square radius around center cell
    """
    
    def __init__(self, radius: int = 1):
        """Constructs a new FreezingToCold Layer Object"""
        super().__init__()
        assert radius >= 0
        self.radius = radius
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        next_board = copy.deepcopy(board)
        rows, cols = board.shape
        
        group_a = set([Cell.FREEZING])
        group_b = set([Cell.WARM, Cell.TEMPERATE])
        
        for i in range(rows):
            for j in range(cols):
                curr: Cell = board[i][j]
                if is_edge_cell(board, i, j, group_a, group_b):
                    next_board = set_board_region(board, i, j, 1, Cell.COLD)
                    
        return next_board
