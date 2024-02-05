# 454_FinalProject
SSU - CS 454 - Final Project - Accelerating Random World Generation with Stochastic Cellular Automaton

This project explores how to efficiently increase the random world generation algorithm for biome maps. The base algorithm
is based off of the core algorithms that are used within the videogame Minecraft. In this project, we present to different 
algorithms (naive and "smart") which create identical biome maps from a given random seed integer. With our "smart" algorithm, 
we were able to increase the efficiency by 2.75x on average.

## SETUP
```



```

## Average Stack Sparsity Levels
|Stack Number | Layer Type | Dimensions | Avg. Sparsity | Avg Number of Unnecessary Updates |
| :---: | :---: | :---: | :---: | :---: |
| 0  | Island             | 4x4       | 28.12% | 11.50 |
| 1  | FuzzyZoom          | 8x8       | 13.75% | 55.20 |
| 2  | AddIsland          | 8x8       | 20.00% | 51.20 |
| 3  | Zoom               | 16x16     | 13.98% | 220.21 |
| 4  | AddIsland          | 16x16     | 18.28% | 209.20 |
| 5  | AddIsland          | 16x16     | 21.48% | 201.01 |
| 6  | AddIsland          | 16x16     | 25.08% | 191.80 |
| 7  | RemoveTooMuchOcean | 16x16     | 85.94% | 35.99 |
| 8  | AddTemps           | 16x16     | 88.87% | 28.49 |
| 9  | AddIsland          | 16x16     | 92.77% | 18.51 |
| 10 | WarmToTemperate    | 16x16     | 90.04% | 25.50 |
| 11 | FreezingToCold     | 16x16     | 90.27% | 24.91 |
| 12 | Zoom               | 32x32     | 69.82% | 309.04 |
| 13 | Zoom               | 64x64     | 47.82% | 2137.29 |
| 14 | AddIsland          | 64x64     | 60.22% | 1629.39 |
| 15 | TemperatureToBiome | 64x64     | 82.71% | 708.20 |
| 16 | Zoom               | 128x128   | 69.76% | 4954.52 |
| 17 | Zoom               | 256x256   | 50.97% | 32132.30 |
| 18 | Zoom               | 512x512   | 34.81% | 170891.67 |
| 19 | AddIsland          | 512x512   | 40.43% | 156159.18 |
| 20 | Zoom               | 1024x1024 | 28.30% | 751828.99 |

