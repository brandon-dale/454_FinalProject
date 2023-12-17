import numpy as np
from core import Cell, get_neighbors, Board
from lazy_layers.layer import Layer

class SmartAddIsland(Layer):
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
        read_ind = board.current_board_idx
        set_ind = board.next_board_idx
        
        def is_edge_cell(r, c):
            if board.board[r][c][read_ind] != Cell.OCEAN:
                # Check adjacent cells for ocean
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < dims and 0 <= nc < dims and board.board[nr][nc][read_ind] == Cell.OCEAN:
                        return True
            elif board.board[r][c][read_ind] == Cell.OCEAN:
                # Check adjacent cells for land
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < dims and 0 <= nc < dims and board.board[nr][nc][read_ind] != Cell.OCEAN:
                        return True
            return False
        
        # If edge map exists (SHOULD EXIST)
        if board.edge_map is not None:
            for r, c in board.edge_map.map:
                if is_edge_cell(r, c):
                    becomes_land: bool = rng.uniform(0.0, 1.0) < SmartAddIsland.P_LAND
                    
                    # ocean -> land
                    if board.board[r][c][read_ind] == Cell.OCEAN and becomes_land:
                        neighbors = get_neighbors(board, r, c, 1).flatten()
                        index = np.argwhere(neighbors == Cell.OCEAN)
                        neighbors = np.delete(neighbors, index)
                        neighbor = rng.choice(neighbors)
                        board.board[r][c][set_ind] = neighbor
                    # land -> ocean
                    elif board.board[r][c][read_ind] != Cell.OCEAN and not becomes_land:
                        board.board[r][c][set_ind] = Cell.OCEAN
        else:
            raise NotImplementedError
            
        board.swap_buffers()


