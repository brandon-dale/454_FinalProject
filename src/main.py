import numpy as np
 
from core import np_random, draw, Cell
from lazy_layers.Island import Island
from lazy_layers.Zoom import Zoom
from lazy_layers.FuzzyZoom import FuzzyZoom


def build(rng: np.random.Generator) -> np.ndarray:
    # specify the build stack
    island_layer = Island()
    stack = [
        FuzzyZoom()
    ]
    
    # Run the stack
    print(f"Running Layer : {island_layer.__class__.__name__}")
    board: np.ndarray = island_layer.run(4, rng)
    draw(board, 'init', 'init')
    
    for layer in stack:
        print(f"Running Layer : {layer.__class__.__name__}")
        board = layer.run(board, rng)
        draw(board, 'fuzzy-zoom', 'fuzzy-zoom')
    
    return board


def main():
    rng, np_seed = np_random(seed=0)
    
    board = build(rng)
    
    draw(board, "main board", 'main_board')
    

if __name__ == '__main__':
    main()