# ExperimentData
Public data from experiments


## Folder "results/"

Contains result files for all 100 runs per game, for all tuners and RHEA configuration combinations. 
- A "-500" flag on directory name means the budget used for the run was 500 forward model calls. A "-5000" flag indicates a budget of 5000 FM calls. If no budget flag is used, the default budget is used, 1000 FM calls. 
- A "-15" flag indicates RHEA configuration is 10-15 (population size - individual length), as opposed to 5-10 if the flag is missing. 

## Folder "py/"

Contains python scripts used to process all the results and produce table and plot data.

## Folder "plots/"

Contains all plots generated, some of which are included in the paper.
