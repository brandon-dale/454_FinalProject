import numpy as np
from lazy_layers.layer import Layer
import copy

class TemperatureToBiome(Layer):
    """
    Processing layer in world generation.
    """
    # BIOME_ODDS = (
    #     ([0, 1](0.7, 0.3)),
    #     ([0, 1], (0.5, 0.5)),
    #     ([0, 1, 2], (0.5, 0.25, 0.25)),
    #     ([0, 1, 2, 3, 4], (0.3, 0.3, 0.2, 0.1, 0.1))
    # )
    
    def __init__(self):
        """Constructs a new TemperatureToBiome Layer Object"""
    
    def run(self, board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :return: A new copy of the board after the transformation
        """
        n_rows, n_cols = board.shape
        
        # for i in range(n_rows):
        #     for j in range(n_cols):
        #         vals, odds = TemperatureToBiome.BIOME_ODDS[board[i][j]]
        #         offset = rng.choice(vals, p=odds)
        #         board[i][j] = board[i][j] * 10 + offset
        
        return board


# Warm:
# 30% Desert
# 30% Plains
# 20% Rainforest
# 10% Savannah
# 10% Swamp
# (0.3, 0.3, 0.2, 0.1, 0.1)

# Temperate:
# 50% Woodland
# 25% Forest
# 25% Highland
# (0.5, 0.25, 0.25)

# Cold:
# 50% Taiga
# 50% Snowy Forest
# (0.5, 0.5)

# Freezing:
# 70% Tundra
# 30% Ice Plains
# (0.7, 0.3)