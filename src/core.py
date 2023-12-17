from typing import Optional, Tuple, Any, Set
import numpy as np
import matplotlib.pyplot as plt
import enum
import glob
import os
import contextlib
from PIL import Image


class Cell(enum.IntEnum):
    """
    Stores the possible contents of a single cell in a board.
    NOTE: Temperature -> Biome conversion
      Multiply the temperature by 10 and randomly add some offset
      to get the new biome.
    """
    # ------ Land vs Ocean ------ #
    OCEAN = 0
    LAND = 1
    # ------ Temperature ------ #
    FREEZING = 2
    COLD = 3
    TEMPERATE = 4
    WARM = 5
    # ------ Biomes ------ #
    # FREEZING
    TUNDRA = 20
    ICE_PLAINS = 21
    # COLD
    TAIGA = 30
    SNOWY_FOREST = 31
    # TEMPERATE
    WOODLAND = 40
    FOREST = 41
    HIGHLAND = 42
    # WARM
    DESERT = 50
    PLAINS = 51
    RAINFOREST = 52
    SAVANNAH = 53
    SWAMP = 54
    # ------ SPECIAL CASES ------ #
    SHORE = 60
    SWAMP_SHORE = 61
    DEEP_OCEAN = 62
    SHALLOW_OCEAN = 63
    NULL = -1 
    
    def __repr__(self):
        return str(self.value)
    

def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, Any]:
    """Generates a random number generator from the seed and returns the Generator and seed.

    Args:
        seed: The seed used to create the generator

    Returns:
        The generator and resulting seed

    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


# map cell states to RGB colors
_COLOR_MAP = {
    # ------ Land vs Ocean ------ #
    Cell.OCEAN: (0,0,255),
    Cell.LAND: 	(255,250,205),
    # ------ Temperature ------ #
    # Cell.FREEZING: (216, 255, 255),
    # Cell.COLD: (3, 127, 81),
    # Cell.TEMPERATE: (255, 204, 153),
    # Cell.WARM: (193, 168, 107),
    Cell.FREEZING: (255, 255, 255),
    Cell.COLD: (127, 127, 127),
    Cell.TEMPERATE: (0, 255, 0),
    Cell.WARM: (255, 0, 0),
    # ------ Biomes ------ #
    # FREEZING
    Cell.TUNDRA: (153, 255, 255),
    Cell.ICE_PLAINS: (204, 255, 255),
    # COLD
    Cell.TAIGA: (204, 229, 255),
    Cell.SNOWY_FOREST: (153, 204, 255),
    # TEMPERATE
    Cell.WOODLAND: (76, 153, 0),
    Cell.FOREST: (0, 153, 0),
    Cell.HIGHLAND: (0, 204, 102),
    # WARM
    Cell.DESERT: (255, 255, 153),
    Cell.PLAINS: (0, 204, 102),
    Cell.RAINFOREST: (0, 102, 51),
    Cell.SAVANNAH: (204, 204, 0),
    Cell.SWAMP: (76, 153, 0),
    # ------ SPECIAL CASES ------ #
    Cell.SHORE: (255, 255, 204),
    Cell.SWAMP_SHORE: (153, 153, 0),
    Cell.DEEP_OCEAN: (0, 51, 102),
    Cell.SHALLOW_OCEAN: (102, 178, 255)
}


def draw(board: Any, title: str, out_file: str=None) -> None:
    """
    Draws the board
    :param board: the board the draw (np.ndarray or Board object)
    :param title: the name of the title for the plot
    :param out_file: optional, if set, an output file will be created
                     instead of drawing in interactive window

    """
    # Expand board to RGB image
    if isinstance(board, np.ndarray):
        assert board.shape[0] == board.shape[1], "CANNOT DRAW A NON-SQUARE BOARD"
        dim = board.shape[0]
        colors = np.zeros((board.shape + (3,)), dtype=int)
        for i in range(dim):
            for j in range(dim):
                r, g, b = _COLOR_MAP[board[i, j]]
                colors[i, j, 0] = r
                colors[i, j, 1] = g
                colors[i, j, 2] = b
    elif isinstance(board, Board):
        dim = board.dims
        colors = np.zeros(((dim, dim) + (3,)), dtype=int)
        for i in range(dim):
            for j in range(dim):
                r, g, b = _COLOR_MAP[board.board[i][j][board.current_board_idx]]
                colors[i, j, 0] = r
                colors[i, j, 1] = g
                colors[i, j, 2] = b

    plt.imshow(colors, extent=[0, dim, 0, dim])
    ax = plt.gca()
    ticks = np.arange(0, dim + 1, int(dim/4.))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.title(title)
    
    if out_file is not None:
        plt.savefig('src/imgs/' + out_file)
    else:
        plt.show()
    

def make_gif(img_dir_path: str, out_file_path: str) -> None:
    # Get the paths sorted
    paths = glob.glob(img_dir_path + '/*.png')
    paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    
    # Make the GIF
    frames = [Image.open(image) for image in paths]
    frame_one = frames[0]
    frame_one.save(out_file_path + ".gif", format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)


def in_bounds(board: np.ndarray, row_ind: int, col_ind: int):
    """
    Check if the cell at board[row_ind][col_ind] is in the bounds of the board
    :param board: a 2d board to check against
    :param row_ind: a valid row index to use
    :param col_ind: a valid column index to use
    :return: true if board[row_ind][col_ind] is in the bounds of the board
    """
    num_rows, num_cols = board.shape
    return row_ind >= 0 and col_ind >= 0 and row_ind < num_rows and col_ind < num_cols


def get_neighbors(board: Any, row_ind: int, col_ind: int, radius: int = 1):
    """
    Returns a copy of the neighbors in a radius of the specified cell on the board
    ex:
    1 2 3 4           1 2 3
    2 3 4 5  (1, 1)   2 3 4
    3 4 5 6  ----->   3 4 5
    4 5 6 7
    """
    if isinstance(board, np.ndarray):
        n_rows, n_cols = board.shape
        i_start = max(0, row_ind-radius)
        i_end = min(row_ind+1+radius, n_rows)
        j_start = max(0, col_ind-radius)
        j_end = min(col_ind+1+radius, n_cols)
        return board[i_start:i_end, j_start:j_end]
    elif isinstance(board, Board):
        dims = board.dims
        i_start = max(0, row_ind-radius)
        i_end = min(row_ind+1+radius, dims)
        j_start = max(0, col_ind-radius)
        j_end = min(col_ind+1+radius, dims)
        return board.board[i_start:i_end, j_start:j_end, board.current_board_idx]
    else:
        raise KeyError


def is_edge_cell(board: np.ndarray, 
                 row_ind: int, 
                 col_ind: int, 
                 group_a: Set[Cell] = None, 
                 group_b: Set[Cell] = None):
    """
    Check if the cell at board[row_ind][col_ind] is an edge cell.
    Is an edge cell if any surrounding cells have a different value
    Only checks up/down and left/right -- no diagonals
    :param board: a 2d board to use
    :param row_ind: a valid row index to use
    :param col_ind: a valid column index to use
    :param group_a: A set of cells considered in one group to ignore
    :param group_b: A set of cells in a second comparison group
    :return: true if board[row_ind][col_ind] is an edge cell    
    """
    n_rows, n_cols = board.shape
    center_val = board[row_ind][col_ind]
    is_edge: bool = False

    if group_a is None:
        group_a = set([center_val])
        group_b = set([e.value for e in Cell if e not in group_a])
    
    if center_val not in group_a and center_val not in group_b:
        return False
    
    # key_set = group_a if center_val in group_a else group_b
    target_set = group_b if center_val in group_a else group_a
    
    # left
    is_edge = is_edge or (col_ind-1 >= 0 and 
                          board[row_ind][col_ind-1] != center_val and 
                          board[row_ind][col_ind-1] in target_set)
    # right
    is_edge = is_edge or (col_ind+1 < n_cols and 
                          board[row_ind][col_ind+1] != center_val and
                          board[row_ind][col_ind+1] in target_set)
    # up
    is_edge = is_edge or (row_ind-1 >= 0 and 
                          board[row_ind-1][col_ind] != center_val and
                          board[row_ind-1][col_ind] in target_set)
    # down
    is_edge = is_edge or (row_ind+1 < n_rows and 
                          board[row_ind+1][col_ind] != center_val and
                          board[row_ind+1][col_ind] in target_set)
    
    return is_edge


def set_board_region(board: np.ndarray, row_ind: int, col_ind: int, radius: int, set_val: Cell):
    """
    Set a region of the board around (row_ind, col_ind) with a radius of radius to
    the specified value
    """
    n_rows, n_cols = board.shape
    i_start = max(0, row_ind-radius)
    i_end = min(row_ind+1+radius, n_rows)
    j_start = max(0, col_ind-radius)
    j_end = min(col_ind+1+radius, n_cols)
    
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            if board[i][j] != Cell.OCEAN:
                board[i][j] = set_val
    
    return board


class EdgeMap:
    """
    Stores the information for a map of edges on a 2D board of cells
    """    
    def __init__(self, dims: int = None, board: np.ndarray = None):
        """
        Initialize a new edge map object.
        Either construct an empty map by specifying dims, or generate
        an edge map from an existing board
        """
        assert dims is not None or board is not None, "Must specify at least edge map dims or a board"
        # Track the dimensions of the board
        self.dims = dims if dims is not None else board.shape[0]
        # Maps 2D index (row, col) -> Cell type where each item is an edge cell
        self.map = {}
        # A list of cells that need to be checked for validity
        self.check = []
        
        if board is None:
            return

        def is_edge_cell(r, c):
            key = board[r][c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.dims and 0 <= nc < self.dims and board[nr][nc] != key:
                    return True
            return False

        # Read in edges from the given board
        for r in range(self.dims):
            for c in range(self.dims):
                if is_edge_cell(r, c):
                    self.map[(r, c)] = board[r][c]
        
    def addEdge(self, r: int, c: int, cellVal: Cell = None, valid: bool = True) -> None:
        """
        Add an edge to the edge map
        Automatically handles edge detection
        If cell already exists, it will overwrite the current value
        :param r: the row index to set
        :param c: the column index to set
        :param cellVal: the value of the cell to set
        """
        # Boundary check
        if cellVal is None or (r < 0) or (r >= self.dims) or (c < 0) or (c >= self.dims):
            return

        # Add edge
        self.map[(r, c)] = cellVal
        
        if not valid:
            self.check.append((r, c))
    
    def scale(self):
        """
        Scales the board by a factor of 2 and updates map
        so that it only contains edge cells
        """
        # Scale the board - not everything may be an edge after
        new_map = {}
        for r, c in [*self.map]:
            key = self.map[(r, c)]
            new_r, new_c = 2 * r, 2 * c
            
            # check up
            up = self.map.get((r-1, c))
            if (up is not None) and (up != key): 
                new_map[(new_r, new_c)] = key
                new_map[(new_r, new_c+1)] = key
            # check down
            down = self.map.get((r+1, c))
            if (down is not None) and (down != key): 
                new_map[(new_r+1, new_c)] = key
                new_map[(new_r+1, new_c+1)] = key
            # check left
            left = self.map.get((r, c-1))
            if (left is not None) and (left != key): 
                new_map[(new_r, new_c)] = key
                new_map[(new_r+1, new_c)] = key
            # check right
            right = self.map.get((r, c+1))
            if (right is not None) and (right != key): 
                new_map[(new_r, new_c+1)] = key
                new_map[(new_r+1, new_c+1)] = key

        self.dims *= 2
        self.map = new_map
                    
    def at(self, r: int, c: int, infer: bool = False):
        """
        Return the value of the cell at (row, col)
        If infer is set to true, it will infer what value should be there.
        If not in bounds or no mapping for that index, returns None if infer is false
        :param r: the row index
        :param c: the column index
        :param infer: if true, will do a weak infer on the cells value if cell value 
                      does not exist. Otherwise, return None
        :return: the cell value if the cell exists, None if cell doesn't exist/isn't set
                 and infer is false, otherwise the inferred value
        """        
        # if ((r < 0) or (r >= self.dims) or (c < 0) or (c >= self.dims)) or (r, c) not in self.map:
        if (r, c) not in self.map:
            return self.infer_weak(r, c) if infer else None
        return self.map[(r, c)]
    
    def infer_weak(self, r: int, c: int):
        """
        Infers the value of a non-edge cell.
        Weak mode -- Only check neighboring cells (including diagonals)
        :param r: the row index
        :param c: the column index
        """
        for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
            val = self.map.get((r + dr, c + dc))
            if val is not None:
                return val
        # No valid value found in neighbors
        raise KeyError
        return None

    def remove_non_edge_cells(self, check_all: bool = True):
        """
        Removes all edges in the map that are not edge cells
        :param check_all: if true, check all cells in map. if false
                          only check cells in check list
        """
        # TEMP REMOVE ME!!!!
        # print(f"---- REMOVING NON EDGE CELLS ({len(self.map)}) ----")
        # orig_size = len(self.map)
        
        if check_all:
            for r, c in [*self.map]:
                neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                vals = [self.at(r, c) for r, c in neighbors]
                vals = np.unique([x for x in vals if x is not None])
                if len(vals) == 1 and self.map[(r, c)] == vals[0]:
                    self.map.pop((r, c))
        else:
            for r, c in self.check:
                neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                vals = [self.at(r, c) for r, c in neighbors]
                vals = np.unique([x for x in vals if x is not None])
                if len(vals) == 1 and self.map[(r, c)] == vals[0]:
                    self.map.pop((r, c))
            self.check = []
        
        # new_size = len(self.map)
        # print(f"---- DONE REMOVE NON EDGE CELLS ({orig_size} -> {new_size}) ----")
        # print(f"    Remove {orig_size - new_size} cells")


    def sparsity(self) -> float:
        """Returns the sparcity of the edge map"""
        return len(self.map) / (self.dims ** 2)

    def draw(self, title, out_file) -> None:
        """
        Draw the edge map, where black is an edge, white is a non-edge
        :param title: the name of the title for the plot
        :param out_file: optional, if set, an output file will be created
                         instead of drawing in interactive window
        """
        # Expand edge map to a full binary board
        board = np.zeros((self.dims, self.dims), dtype=int)
        for i in range(self.dims):
            for j in range(self.dims):
                board[i, j] = 0 if (i, j) in self.map else 255

        plt.imshow(board, extent=[0, self.dims, 0, self.dims], cmap='gray', vmin=0, vmax=255)
        ax = plt.gca()
        ticks = np.arange(0, self.dims + 1, int(self.dims/4.))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        title += f"  ({(self.sparsity()*100):.2f}%)"
        plt.title(title)
        
        if out_file is not None:
            plt.savefig('src/imgs/' + out_file)
        else:
            plt.show()
    

class Board:
    """
    Contains the data for a board, where it contains "two copies"
    Each cell is a 2-tuple to track the current state, and the next state
    Contains two board states, current and next.
        Current is the board at time step t, should only read from this
        Next is the board at time step t + 1, should only set values here
    """
    
    def __init__(self, dims: int, board: np.ndarray = None):
        """
        Initializes a new board object around an existing board
        """
        # Expand the original board
        self.board = np.zeros(((dims, dims) + (2,)), dtype=Cell)
        self.dims = dims

        # Tracks which board is the current board and next board
        self.current_board_idx = 0
        self.next_board_idx = 1
        self.edge_map = None # empty edge map
        
        # read in the board if it exists
        if board is not None:
            self.dims = board.shape[0]
            for r in range(self.dims):
                for c in range(self.dims):
                    self.board[r][c][self.current_board_idx] = board[r][c]
    
    def swap_buffers(self):
        """
        Changes the board that is currently active
        Should be done after some update rule has been applied
        to the whole board.
        """
        self.current_board_idx = 1 - self.current_board_idx
        self.next_board_idx = 1 - self.next_board_idx
    
    def set_cell(self, r: int, c: int, val: Cell):
        """
        Sets the value of the cell in the next board state.
        Does not alter the current board's state
        DOES NOT DO BOUNDS CHECKING
        """
        self.board[r][c][self.next_board_idx] = val
        
    def get(self, r: int, c: int):
        """
        Gets the value of the cell in the current board state.
        Returns None on out of bounds
        """
        return self.board[r][c][self.current_board_idx] if 0 <= r < self.dims and 0 <= c < self.dims else None

    def scale(self):
        """scales the board by a factor of 2"""
        self.board = np.repeat(np.repeat(self.board, 2, axis=0), 2, axis=1)
        
        curr_ind = self.current_board_idx
        next_ind = self.next_board_idx
        self.dims *= 2
        
        for r in range(self.dims):
            for c in range(self.dims):
                self.board[r][c][next_ind] = self.board[r][c][curr_ind]
        
            
        if self.edge_map is not None:
            self.edge_map.scale()
    
    def is_edge_cell(self, r: int, c: int):
        """Assumes r, c are valid in board"""
        key = self.board[r][c][self.current_board_idx]
        # Check adjacent cells for ocean
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.dims and 0 <= nc < self.dims and self.board[nr][nc][self.current_board_idx] != key:
                return True
        return False

    def activate_edge_map(self):
        if self.edge_map is None:
            self.edge_map = EdgeMap(self.dims)

    def draw(self, title: str, out_file: str=None):
        """
        draws board
        Copy of regular drawing function for ndarray
        Please don't read this Ravi
        """
        dim = self.dims
        colors = np.zeros(((dim, dim) + (3,)), dtype=int)
        for i in range(dim):
            for j in range(dim):
                r, g, b = _COLOR_MAP[self.board[i][j][self.current_board_idx]]
                colors[i, j, 0] = r
                colors[i, j, 1] = g
                colors[i, j, 2] = b

        plt.imshow(colors, extent=[0, dim, 0, dim])
        ax = plt.gca()
        ticks = np.arange(0, dim + 1, int(dim/4.))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.title(title)
        
        if out_file is not None:
            plt.savefig('src/imgs/' + out_file)
        else:
            plt.show()

