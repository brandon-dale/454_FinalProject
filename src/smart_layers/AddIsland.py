import numpy as np
from core import Cell, get_neighbors, Board
from lazy_layers.layer import Layer

class AddIsland(Layer):
    """
    Processing layer in world generation.
    Each edge cell between land and ocean has a 60% chance of becoming land and a 40% chance of becoming ocean.
    """
    P_LAND = 0.6
    
    def __init__(self):
        """Constructs a new Smart AddIsland Layer Object"""
    
    def run(self, board: Board, rng: np.random.Generator) -> None:
        """
        Runs the layer for a single step
        :param board: A 2D array containing the input board state
        :param rng: A random number generator
        :return: A new copy of the board after the transformation
        """
        dims = board.dims

        # Function to determine if a cell is an edge between land and ocean
        def is_edge_cell(r, c):
            if board[r][c] != Cell.OCEAN:
                # Check adjacent cells for ocean
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == Cell.OCEAN:
                        return True
            elif board[r][c] == Cell.OCEAN:
                # Check adjacent cells for land
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != Cell.OCEAN:
                        return True
            return False

        # Iterating over the edge cells
        for i in range(dims):
            for j in range(dims):
                if is_edge_cell(i, j):
                    neighbors = get_neighbors(board, i, j, 1).flatten()
                    index = np.argwhere(neighbors == Cell.OCEAN)
                    neighbors = np.delete(neighbors, index)
                    neighbor = rng.choice(neighbors)
                    next_board[i][j] = neighbor if rng.uniform(0.0, 1.0) < AddIsland.P_LAND else Cell.OCEAN

        