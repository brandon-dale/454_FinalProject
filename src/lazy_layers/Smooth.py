import numpy as np
from lazy_layers.layer import Layer
from core import Cell, get_neighbors

class Smooth(Layer):
    """
    Processing layer in world generation.
    Uses gaussian smoothing
    """
    
    def __init__(self):
        """Constructs a new Smooth Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        n_rows, n_cols = board.shape
        SEARCH_RAD = 2
        # THRESHOLD = 2
        
        for i in range(n_rows):
            for j in range(n_cols):
                neighbors = get_neighbors(board, i, j, SEARCH_RAD).flatten()
                unique, counts = np.unique(neighbors, return_counts=True)
                count_map = dict(zip(unique, counts))
                
                if Cell.OCEAN in count_map:
                    num_ocean = count_map[Cell.OCEAN]
                    count_map.pop(Cell.OCEAN)
                else:
                    num_ocean = 0
                num_land = sum(count_map.values())
                
                if num_ocean >= num_land:
                    board[i][j] = Cell.OCEAN
                else:
                    index = np.argwhere(unique == Cell.OCEAN)
                    neighbors = np.delete(neighbors, index)
                    neighbor = rng.choice(neighbors)
                    board[i][j] = neighbor
        
        return board
