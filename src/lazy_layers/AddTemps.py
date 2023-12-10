import numpy as np
from core import Cell
from lazy_layers.layer import Layer
import copy

class AddTemps(Layer):
    """
    Processing layer in world generation.
    Each edge cell between land and ocean has a 75% chance of becoming land and a 25% chance of becoming ocean.
    """
    
    def __init__(self):
        """Constructs a new AddIsland Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == Cell.OCEAN:
                    continue
                # Generate a random number to determine the temperature
                temp = rng.integers(1, 7)  # 1-6 range, where 1-4 are warm, 5 is cold, 6 is freezing
                if temp <= 4:
                    board[i, j] = Cell.WARM
                elif temp == 5:
                    board[i, j] = Cell.COLD
                else:
                    board[i, j] = Cell.FREEZING
        
        return board