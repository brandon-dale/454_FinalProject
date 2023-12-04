# LAYERS

## ISLAND
Initial island (starting layer)
- Each cell as a 1/10 percentage of becoming land
- Results should be 9/10 ocean and 1/10 land

## FUZZY ZOOM
Scale by a factor of 2 and:

for each edge cell:
- Generate xoff, yoff [-1, 1] uniformly.
- Cell[i][j]<t+t> = Cell[i + xoff][j + yoff]<t>

## ZOOM
Scale by a factor of 2 and:

for each edge cell:
- Generate xoff, yoff [-1, 1] from a gaussian distribution
- Cell[i][j]<t+t> = Cell[i + xoff][j + yoff]<t>

## ADD ISLAND
For each edge cell:
- 75% -> land
- 25% -> ocean

I randomly picked these numbers, we can play around with them.

## ADD TEMPERATURES
Each cell has a random temperature (warm, cold, or freezing) in a proportion of 4, 1, and 1 respectively

## REMOVE TOO MUCH OCEAN
All ocean regions surrounded by more ocean have a 50% chance of becoming land.

Only check up/down and left/right.

## WARM -> TEMPERATE
Any warm land adjacent to a cool or freezing region will turn into a temperate one instead.

## FREEZING -> COLD
And any freezing land adjacent to a warm or temperate region will turn cold.

## TEMPERATURE -> BIOME
These temperatures are ultimately the main factor in determining which biome a piece of land will end up being. 

Warm:
- 30% Desert
- 30% Plains
- 20% Rainforest
- 10%  Savannah
- 10%  Swamp

Temperate:
- 50% Woodland
- 25% Forest
- 25% Highland

Cold:
- 50% Taiga
- 50% Snowy Forest

Freezing:
- 70% Tundra
- 30% Ice Plains

## SHORE
Add shore on edge of land and ocean.

Choose depth later, based on testing. For now, depth of 3 both ways.

Special case for swamp shore.

## SMOOTH
Use gaussian smoothing. Figure out details later.

## Voronoi Zoom
Include if we can.




