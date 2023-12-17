import numpy as np
from lazy_layers.layer import Layer
from core import Board


class FuzzyZoom(Layer):
    """
    Processing layer in world generation.
    Scales the board by a factor of 2. For each edge cell, generates xoff, yoff in the range [-1, 1]
    uniformly and updates the cell value.
    """

    def __init__(self):
        """Constructs a new FuzzyZoom Layer Object"""
    
    def run(self, board: Board, rng: np.random.Generator) -> None:
        """
        Runs the layer for a single step.
        :param edge_map: An edge map object with the data on the board
        :param rng: A random number generator.
        :return: A new copy of the board after the transformation.
        """
        # Scale the board by a factor of 2
        board.scale()
        dims = board.board.shape[0]
        read_index = board.current_board_idx
        set_index = board.next_board_idx
              
        for i in range(dims):
            for j in range(dims):
                # Only process edge cells
                if board.is_edge_cell(i, j):
                    # Generate xoff and yoff uniformly from [-1, 0, 1]
                    xoff = rng.integers(-1, 2)
                    yoff = rng.integers(-1, 2)

                    # Update the cell value
                    new_i = max(0, min(dims - 1, i + xoff))
                    new_j = max(0, min(dims - 1, j + yoff))
                    board[i][j][set_index] = board[new_i][new_j][read_index]

        board.swap_buffers()