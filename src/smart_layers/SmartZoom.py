import numpy as np
from lazy_layers.layer import Layer
from core import Board


class SmartZoom(Layer):
    """
    Processing layer in world generation.
    """

    def __init__(self):
        """Constructs a new Smart Zoom Layer Object"""
    
    def run(self, board: Board, rng: np.random.Generator) -> None:
        """
        Runs the layer for a single step.
        :param board: Should be a board object, or numpy ndarray
        :param rng: A random number generator.
        :return: A new copy of the board after the transformation.
        """
        # Scale the board by a factor of 2
        board.scale()
        dims = board.dims
        read_index = board.current_board_idx
        set_index = board.next_board_idx
        
        indexes = [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        
        # First time layer is called
        if board.edge_map is None:
            board.activate_edge_map()
            for i in range(dims):
                for j in range(dims):
                    # Only process edge cells
                    if board.is_edge_cell(i, j):
                        # Generate xoff and yoff uniformly from [-1, 0, 1]
                        roff = rng.choice(indexes)
                        coff = rng.choice(indexes)

                        # Update the cell value
                        new_r = max(0, min(dims - 1, i + roff))
                        new_c = max(0, min(dims - 1, j + coff))
                        new_val = board.board[new_r][new_c][read_index]
                        board.board[i][j][set_index] = new_val
                        board.edge_map.addEdge(i, j, board.get(i, j))
        # Not first time - Edge map is set and is valid
        else:
            # For each edge cell
            edge_queue = []
            for r, c in board.edge_map.map:
                roff = rng.choice(indexes)
                coff = rng.choice(indexes)
                
                # Update the cell value
                new_r = max(0, min(dims - 1, r + roff))
                new_c = max(0, min(dims - 1, c + coff))
                
                orig_val = board.board[r][c][read_index]
                new_val = board.board[new_r][new_c][read_index]
                board.board[r][c][set_index] = new_val
                
                if orig_val != new_val:
                    edge_queue.extend([
                        (r-1, c, board.get(r-1, c)), (r+1, c, board.get(r+1, c)), 
                        (r, c-1, board.get(r, c-1)), (r, c+1, board.get(r, c+1))
                    ])
            
            for r, c, val in edge_queue:
                board.edge_map.addEdge(r, c, val)
            
            board.edge_map.remove_non_edge_cells()
        
        board.swap_buffers()

"""
L O O O
L L O O
L O O O
L O O O

"""