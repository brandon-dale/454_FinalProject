import numpy as np
from lazy_layers.layer import Layer

class FuzzyZoom(Layer):
    """
    Processing layer in world generation.    
    Scale by a factor of 2 and for each edge cell:
        Generate xoff, yoff [-1, 1] uniformly.
        Cell[i][j]<t+t> = Cell[i + xoff][j + yoff]
    """
    
    def __init__(self):
        """Constructs a new FuzzyZoom Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        scaled_board = np.repeat(np.repeat(board, 2, axis=0), 2, axis=1)
        
        rows, cols = scaled_board.shape
        
        for i in range(rows):
            for j in range(cols):
                
                # Get random offsets
                xoff, yoff = rng.integers(-1, 2, 2)
                
                 # Update the cell value
                new_i = max(0, min(rows - 1, i + xoff))
                new_j = max(0, min(cols - 1, j + yoff))
                scaled_board[i][j] = scaled_board[new_i][new_j]
        
        return scaled_board