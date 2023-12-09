import numpy as np
 
from core import np_random, draw, Cell
from lazy_layers.AddIsland import AddIsland
from lazy_layers.AddTemps import AddTemps
from lazy_layers.FreezingToCold import FreezingToCold
from lazy_layers.FuzzyZoom import FuzzyZoom
from lazy_layers.Island import Island
from lazy_layers.RemoveTooMuchOcean import RemoveTooMuchOcean
from lazy_layers.Shore import Shore
from lazy_layers.Smooth import Smooth
from lazy_layers.TemperatureToBiome import TemperatureToBiome
from lazy_layers.WarmToTemperate import WarmToTemperate
from lazy_layers.Zoom import Zoom


def build(rng: np.random.Generator) -> np.ndarray:
    # specify the build stack
    island_layer = Island()
    stack = [
        FuzzyZoom(),
        AddIsland(),
        Zoom(),
        AddIsland(),
        AddIsland(),
        AddIsland(),
        RemoveTooMuchOcean()
        # AddTemps()
        # 
    ]
    
    # Run the stack
    board = None
    for i, layer in enumerate(stack):
        title = str(layer.__class__.__name__) + f"_{i}"
        if i == 0:
            print(f"Running Layer : {island_layer.__class__.__name__}")
            board: np.ndarray = island_layer.run(4, rng)
            draw(board, title, title)
            
        print(f"Running Layer : {layer.__class__.__name__} {i}")
        board = layer.run(board, rng)
        draw(board, title, title)
    
    return board


def main():
    rng, np_seed = np_random(seed=0)
    
    board = build(rng)
    
    draw(board, "main board", 'main_board')
    

if __name__ == '__main__':
    main()