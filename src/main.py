import numpy as np
import time
from matplotlib import pyplot as plt

from core import np_random, draw, make_gif, Board
from lazy_layers.AddIsland import AddIsland
from lazy_layers.AddTemps import AddTemps
from lazy_layers.FreezingToCold import FreezingToCold
from lazy_layers.FuzzyZoom import FuzzyZoom
from lazy_layers.Island import Island
from lazy_layers.RemoveTooMuchOcean import RemoveTooMuchOcean
from lazy_layers.Shore import Shore
from lazy_layers.TemperatureToBiome import TemperatureToBiome
from lazy_layers.WarmToTemperate import WarmToTemperate
from lazy_layers.Zoom import Zoom

from smart_layers.SmartZoom import SmartZoom
from smart_layers.SmartAddIsland import SmartAddIsland


def lazy_build(rng: np.random.Generator, drawBoards: bool = False, verbose: bool=False) -> np.ndarray:
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
    ]
    
    # ----- Run the stack ----- #
    if verbose:
        print(f"\n----- RUNNING BUILD STACK -----")
    start_time = time.time()
    board = None
    for i, layer in enumerate(stack):
        # Run Layer
        if verbose:
            print(f"Running Layer : {layer.__class__.__name__} {i}")
        if i == 0:
            board: np.ndarray = layer.run(4, rng)
        else:
            board = layer.run(board, rng)
        
        # Draw Board
        if drawBoards:
            n, m = board.shape
            title = f"({i}) - {layer.__class__.__name__} - ({n}, {m})"
            file_name = f"{i}_" + str(layer.__class__.__name__)
            draw(board, title, file_name)
    
    if verbose:
        print("\n--- %s seconds ---" % (time.time() - start_time), end="\n\n")
    return board


def smart_build(rng: np.random.Generator, drawBoards: bool = False, verbose: bool = False) -> np.ndarray:
    # specify the build stack
    dumb_stack = [
        Island(),
        FuzzyZoom(),
        AddIsland(),
        Zoom(),
        AddIsland(),
        AddIsland(),
        AddIsland(),
        RemoveTooMuchOcean(),
        AddTemps(),
        AddIsland(),
        WarmToTemperate(1),
        FreezingToCold(1),
        Zoom(),
        Zoom(),
        AddIsland(),
        TemperatureToBiome(),
    ]
    smart_stack = [
        SmartZoom(),
        SmartZoom(),
        SmartZoom(),
        SmartAddIsland(),
        SmartZoom(),
    ]
    
    # ----- Run the stack ----- #
    if verbose:
        print(f"\n----- RUNNING DUMB BUILD STACK -----")
    start_time = time.time()
    board = None
    layer_num = 0
    for i, layer in enumerate(dumb_stack):
        layer_num = i
        # Run Layer
        if verbose:
            print(f"Running Layer : {layer.__class__.__name__} {i}")
        if i == 0:
            board: np.ndarray = layer.run(4, rng)
        else:
            board = layer.run(board, rng)
        
        # Draw Board
        if drawBoards:
            n, m = board.shape
            title = f"({layer_num}) - {layer.__class__.__name__} - ({n}, {m})"
            file_name = f"{i}_" + str(layer.__class__.__name__)
            draw(board, title, file_name) # draw ndarray
    
    if verbose:
        print(f"\n----- RUNNING SMART BUILD STACK -----")
    
    board = Board(board.shape[0], board)
    for i, layer in enumerate(smart_stack):
        layer_num += 1
        if verbose:
            print(f"Running Layer : {layer.__class__.__name__} {layer_num}")
        layer.run(board, rng)
        
        # Draw Board
        if drawBoards:
            n = board.dims
            title = f"({layer_num}) - {layer.__class__.__name__} - ({n}, {n})"
            file_name = f"{layer_num}_" + str(layer.__class__.__name__)
            board.draw(title, file_name) # draw board
    
    if verbose:
        print("\n--- %s seconds ---" % (time.time() - start_time), end="\n\n")
    return board


def test_sparsity(iters: int, base_rng: np.random.Generator):
    """
    Get the mean sparcity of iters iterations of randomly generated boards
    """
    # Specify the build stack
    stack = [
        Island(),
        FuzzyZoom(),
        AddIsland(),
        Zoom(),
        AddIsland(),
        AddIsland(),
        AddIsland(),
        RemoveTooMuchOcean(),
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


def speed_test(iters, rng: np.random.Generator, builder):
    """
    Run an average speed test on the specified builder function
    """
    avg_speed = 0.
    for _ in range(iters):
        start_time = time.time()
        board = builder(rng, drawBoards=False)
        ex_time = time.time() - start_time
        avg_speed += ex_time
    avg_speed /= iters
    print(f"Average Speed: {avg_speed}")
    
    
def main():
    SEED = 849474937830
    rng, _ = np_random(seed=SEED)
    
    # Build the lazy board :(
    board = lazy_build(rng, True)    
    
    # Build the smart board
    # board = smart_build(rng, drawBoards=True)
    
    make_gif('src/imgs', f"lazy_gif_{SEED}")
    
    
if __name__ == '__main__':
    main()
