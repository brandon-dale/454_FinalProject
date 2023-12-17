import numpy as np
import time
from matplotlib import pyplot as plt
 
from core import np_random, draw, Cell, make_gif, EdgeMap
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

from smart_layers.Island import Smart_Island


def build(rng: np.random.Generator, drawBoards: bool = False) -> np.ndarray:
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
        # Zoom(), # PUT THIS LAYER BACK IN AFTER SPARSITY TESTING
        # Zoom()
    ]
    
    # ----- Run the stack ----- #
    print(f"\n----- RUNNING BUILD STACK -----")
    start_time = time.time()
    board = None
    for i, layer in enumerate(stack):
        # Run Layer
        print(f"Running Layer : {layer.__class__.__name__} {i}")
        if i == 0:
            board: np.ndarray = layer.run(4, rng)
        else:
            board = layer.run(board, rng)
        
        # Draw Board
        if drawBoards:
            n, m = board.shape
            title = f"{layer.__class__.__name__} - ({n}, {m})"
            file_name = f"{i}_" + str(layer.__class__.__name__)
            draw(board, title, file_name)
    
    print("\n--- %s seconds ---" % (time.time() - start_time), end="\n\n")
    return board


def test_sparsity(iters: int, base_rng: np.random.Generator):
    """Get the mean sparcity of iters iterations of randomly generated boards"""
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
        # Zoom(), # PUT THIS LAYER BACK IN AFTER SPARSITY TESTING
        # Zoom()
    ]
    
    # Setup the history dict -- idx -> (mean_sparsity, layer_type)
    history = {i: [0, layer.__class__.__name__] for i, layer in enumerate(stack)}
    
    # Define a sparsity helper functions
    def get_sparsity():
        num_edges = 0
        dims = board.shape[0]
        for r in range(dims):
            for c in range(dims):
                # Test cell
                key = board[r][c]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr >= 0) and (nc >= 0) and (nr < dims) and (nc < dims) and (board[nr][nc] != key): # invalid cell
                        num_edges += 1
                        break
        return num_edges / (dims ** 2)
    
    # Run each test
    print(f"------ TESTING AVG SPARSITY OF STACK OVER {iters} ITERATIONS ------")
    for iter in range(iters):
        print(f"Running Iteration {iter}")
        # Make a new random number generator 
        new_seed = base_rng.integers(0, 999999999999)
        rng, _ = np_random(seed=new_seed)
        
        board = None
        for i, layer in enumerate(stack):
            if i == 0:
                board: np.ndarray = layer.run(4, rng)
            else:
                board = layer.run(board, rng)
            
            sparsity = get_sparsity()
            history[i][0] += sparsity
        #     print(f"    Layer: {i:2} - {history[i][1]:10} - {sparsity*100:.2f}%")
        # print("\n\n")
    
    # Divide each sum to get average
    for key in history:
        history[key][0] /= iters
    
    # Print out the data
    print("---- SUMMARY ----")
    for key in history:
        sparsity, name = history[key]
        print(f"Layer: {key:2} - {name:10} - {sparsity*100:.2f}%")
        
    # Make graph
    x = range(len(stack))
    y = [history[key][0] for key in history]
    plt.plot(x, y)
    plt.title('Average Board Sparcity Through Build Stack')
    plt.ylim((0.0, 1.0))
    plt.show()
    
    return history



def pretty_print_map(_map):
    print("---- MAPPING ----")
    for r, c in _map:
        print(f"({r:2}, {c:2}) --> {_map[(r, c)]}")
    print('\n-----------------\n\n')


def main():
    SEED = 473443303
    rng, _ = np_random(seed=SEED)
    
    # edge_map = EdgeMap(4)
    # edges = [
    #     (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 1), (1, 2, 1), (1, 3, 0), (2, 0, 0), (2, 1, 1),
    #     (2, 2, 0), (3, 1, 0)
    # ]
    # for r, c, val in edges:
    #     edge_map.addEdge(r, c, val)
    # pretty_print_map(edge_map.map)
    # edge_map.scale()
    # pretty_print_map(edge_map.map)
    
    # island = Smart_Island()
    # edge_map: EdgeMap = island.run(4, rng)
    # pretty_print_map(edge_map.map)
    # edge_map.draw("Basic Edge Map", "edge_map_test_1")
    
    board = build(rng, drawBoards=True)
    draw(board, "main board", 'main_board')
    
    # make_gif('src/imgs', f"Map_Generation_Lazy_{SEED}.gif")

    

if __name__ == '__main__':
    main()
