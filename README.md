# ExperimentData
Public data from experiments for paper:

Raluca D. Gaina, Simon M. Lucas, Diego Perez-Liebana, “Tackling Sparse Rewards in Real-Time Games with Statistical Forward Planning Methods”, AAAI Conference on Artificial Intelligence (AAAI-19), 2019, accepted.

https://rdgain.github.io/publications

File format:

(true if rhea : false)\_(rollout depth)\_(population size)\_(FM call budget)\_(heuristic)\_(true if dynamic rollout depth : false)

Data from 3 experiments:
- Score heuristic function vs Win/Lose only
- Extreme length rollout with budget increased proprtionally (lengths 50, 100, 150, 200)
- Dynamic rollout length adjustment based on fitness landscape flatness
