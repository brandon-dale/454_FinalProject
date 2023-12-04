from typing import Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import enum


class cell:
    OCEAN = 0
    LAND = 1


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
    cell.OCEAN: (30,144,255),
    cell.LAND: 	(255,250,205)
}


def draw(board: np.ndarray) -> None:
    """
    Draws the board
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
    plt.show()
    