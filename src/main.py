import numpy as np
from PIL import Image
import time
 
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
    stack = [
        Island(),
        FuzzyZoom(),
        AddIsland(),
        Zoom(),
        AddIsland(),
        AddIsland(),
        AddIsland(),
        RemoveTooMuchOcean(),
        # Smooth(),
        AddTemps(),
        AddIsland(),
        WarmToTemperate(1),
        FreezingToCold(1),
        Zoom(),
        Zoom(),
        AddIsland(),
        TemperatureToBiome(),
        Zoom(),
        Zoom(),
        Zoom(),
        AddIsland(),
        Zoom(),
        # Shore(),
        Zoom(),
        # Zoom()
    ]
    
    # Run the stack
    board = None
    for i, layer in enumerate(stack):
        title = f"{i}_" + str(layer.__class__.__name__)
        
        if i == 0:
            print(f"Running Layer : {layer.__class__.__name__}")
            board: np.ndarray = layer.run(4, rng)
            draw(board, title, title)
        else:
            print(f"Running Layer : {layer.__class__.__name__} {i}")
            board = layer.run(board, rng)
            draw(board, title, title)
    
    return board


def main():
    SEED = 473443303
    
    rng, np_seed = np_random(seed=SEED)
    
    start_time = time.time()
    
    board = build(rng)
    
    # im = Image.fromarray(board)
    # im.save(f"final_map_{SEED}.jpeg")
    
    print("\n--- %s seconds ---" % (time.time() - start_time), end="\n\n")
    
    # draw(board, "main board", 'main_board')
    

if __name__ == '__main__':
    main()
