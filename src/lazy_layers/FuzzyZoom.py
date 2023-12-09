import numpy as np
from lazy_layers.layer import Layer

class FuzzyZoom(Layer):
    """
    Processing layer in world generation.
    Scales the board by a factor of 2. For each edge cell, generates xoff, yoff in the range [-1, 1]
    uniformly and updates the cell value.
    """

    def __init__(self):
        """Constructs a new FuzzyZoom Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step.
        :param board: A 2D array containing the input board state.
        :param rng: A random number generator.
        :return: A new copy of the board after the transformation.
        """
        # Scale the board by a factor of 2
        scaled_board = np.repeat(np.repeat(board, 2, axis=0), 2, axis=1)
     
        rows, cols = scaled_board.shape

        for i in range(rows):
            for j in range(cols):
                # Only process edge cells
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    # Generate xoff and yoff uniformly from [-1, 0, 1]
                    xoff = rng.integers(-1, 2)
                    yoff = rng.integers(-1, 2)

                    # Update the cell value
                    new_i = max(0, min(rows - 1, i + xoff))
                    new_j = max(0, min(cols - 1, j + yoff))
                    scaled_board[i][j] = scaled_board[new_i][new_j]

        return scaled_board
