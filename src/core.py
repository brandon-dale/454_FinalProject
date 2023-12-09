from typing import Optional, Tuple, Any, Set
import numpy as np
import matplotlib.pyplot as plt
import enum


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
    Cell.OCEAN: (30,144,255),
    Cell.LAND: 	(255,250,205),
    # ------ Temperature ------ #
    Cell.FREEZING: (0, 76, 153),
    Cell.COLD: (0, 102, 204),
    Cell.TEMPERATE: (255, 204, 153),
    Cell.WARM: (255, 128, 0),
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


def draw(board: np.ndarray, title: str, out_file: str=None) -> None:
    """
    Draws the board
    :param board: the board the draw
    :param title: the name of the title for the plot
    :param out_file: optional, if set, an output file will be created
                     instead of drawing in interactive window

    """
    # Expand board to RGB image
    assert board.shape[0] == board.shape[1], "CANNOT DRAW A NON-SQUARE BOARD"
    dim = board.shape[0]
    colors = np.zeros((board.shape + (3,)), dtype=int)
    for i in range(dim):
        for j in range(dim):
            r, g, b = _COLOR_MAP[board[i, j]]
            colors[i, j, 0] = r
            colors[i, j, 1] = g
            colors[i, j, 2] = b

    plt.imshow(colors, extent=[0, dim, 0, dim])
    ax = plt.gca()
    ticks = np.arange(0, dim + 1, int(dim/4.))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.title(title)
    
    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    

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


def get_neighbors(board: np.ndarray, row_ind: int, col_ind: int, radius: int = 1):
    """
    Returns a copy of the neighbors in a radius of the specified cell on the board
    ex:
    1 2 3 4           1 2 3
    2 3 4 5  (1, 1)   2 3 4
    3 4 5 6  ----->   3 4 5
    4 5 6 7
    """
    n_rows, n_cols = board.shape
    i_start = max(0, row_ind-radius)
    i_end = min(row_ind+1+radius, n_rows)
    j_start = max(0, col_ind-radius)
    j_end = min(col_ind+1+radius, n_cols)
    return board[i_start:i_end, j_start:j_end]


def is_edge_cell(board: np.ndarray, row_ind: int, col_ind: int, allowed_cells: Set[Cell] = None):
    """
    Check if the cell at board[row_ind][col_ind] is an edge cell.
    Is an edge cell if any surrounding cells have a different value
    Only checks up/down and left/right -- no diagonals
    :param board: a 2d board to use
    :param row_ind: a valid row index to use
    :param col_ind: a valid column index to use
    :param allowed_cells: a list of cells allowed to be considered in edge decisions
                          Defaults to all possible cell values
    :return: true if board[row_ind][col_ind] is an edge cell    
    """
    if allowed_cells is None:
        allowed_cells = set([e.value for e in Cell])
    
    n_rows, n_cols = board.shape
    center_val = board[row_ind][col_ind]
    is_edge: bool = False
    
    if center_val not in allowed_cells:
        return False
    
    # left
    is_edge = is_edge or (col_ind-1 >= 0 and 
                          board[row_ind][col_ind-1] != center_val and 
                          board[row_ind][col_ind-1] in allowed_cells)
    # right
    is_edge = is_edge or (col_ind+1 < n_cols and 
                          board[row_ind][col_ind+1] != center_val and
                          board[row_ind][col_ind+1] in allowed_cells)
    # up
    is_edge = is_edge or (row_ind-1 >= 0 and 
                          board[row_ind-1][col_ind] != center_val and
                          board[row_ind-1][col_ind] in allowed_cells)
    # down
    is_edge = is_edge or (row_ind+1 < n_rows and 
                          board[row_ind+1][col_ind] != center_val and
                          board[row_ind+1][col_ind] in allowed_cells)
    
    return is_edge


def set_board_region(board: np.ndarray, row_ind: int, col_ind: int, radius: int, set_val: Cell):
    """
    
    """
    raise NotImplementedError