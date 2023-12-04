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
    cell.OCEAN: (0, 0, 255),
    cell.LAND: (0, 255, 0)
}


def draw(board: np.ndarray) -> None:
    """
    Draws the board
    """
    # Convert to colors array
    # new_array = np.zeros((a.shape + (3,)))
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[-1]):
    #         new_array[i, j, 0] = a[i, j][0]
    #         new_array[i, j, 1] = a[i, j][1]
    #         new_array[i, j, 2] = a[i, j][2]
    