import numpy as np
from numpy.random import Generator
from lazy_layers.layer import Layer
from core import Cell, EdgeMap, Board
from typing import Tuple

class Smart_Island(Layer):
    """
    Processing layer in world generation for adding an initial
    set of islands to the map.
    Smart Version
    """
    INIT_DIM = 4
    
    def __init__(self):
        """Constructs a new Smart Island Layer Object"""
    
    def run(self, dims: int, rng: np.random.Generator) -> Board:
        """
        Runs the layer for a single step
        Returns an 
        :param dims: the resulting dimensions of the board
        :param rng: a random number generator to use
        :return: an EdgeMap object containing only the edges of the
                 generated board
        """
        PROB_LAND = 0.1
        
        # Generate the initial board
        board = Board(dims)
        for i in range(dims):
            for j in range(dims):
                board.set_cell(i, j, Cell.LAND if rng.uniform(0.0, 1.0) <= PROB_LAND else Cell.OCEAN)
                 
        board.goto_next_board()       
        return board