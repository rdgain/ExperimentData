# ExperimentData
Public data from experiments for the "Rolling Horizon Evolutionary Algorithms for General Video Game Playing" paper, see pdf for details.


## Results folder

Title format: thyia_{random seed}_{game}_{number NTBEA iterations}_{game change frequency}.txt

In each file:
- Results of all games played for configuration evaluation, format: Result {win/loss} {game score} {game ticks}
- Solution evaluated for games played and its fitness
- Solution considered best after each fitness evaluation
- At the end, 100 runs on the same game with the best solution found
- At the end, 100 runs per each of the 20 games tested

## Py folder

Scripts used to process results and generate plots and result summary. Run *main.py*.

## Plots folders

1tuples: plots of all 1tuples based on the results folder, 1 plot per parameter.
2tuples: plots of all 2tuples based on the results folder, 1 plot per combination of 2 parameters.
fitness: plots of fitness evaluation progression in NTBEA, 1 plot per game.
