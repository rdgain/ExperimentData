# ExpRHEA

**res/rhea-journal/results**

directory names (each value sepated through underscore _):
- "true" (if algorithm is RHEA) or "false" (if algorithm is MCTS)
- rollout length
- population size or analysis window size
- (if RHEA): 0 (if initialization random) or 2 (if initialization MCTS)
- (if RHEA): true (if using shift buffer and MC rollouts at the end of individual evaluation) or false (if not used)

in each directory, file names (each value sepated through underscore _):
- directory name +
- game index (0-101)

in each file, alternating lines (200 total: 2 per run, x 5 levels x 20 times each level) (each value sepated through space):
- Evo +
	- Average convergence
	- Stat summary best fitness value: MIN MAX MEAN Q1 MEDIAN Q3 SD STDERR ENTROPY
	- Percentage level explored in the actual game
	- Percentage level explored in the Forward Model
	- First tick a win was seen
	- First tick a loss was seen
- Result +
	- 1 (if win) or 0 (if lose)
	- game score
	- game ticks
	- entropy over count of actions chosen during the game
	- 6 values: percentage each action was chosen during the game, -1 if not at all

<hr>

**res/rhea-journal/evofiles**
 
file names (each value sepated through underscore _): 
- "evo" (if algorithms is RHEA) or "mcts" (if algorithm is MCTS)
- rollout length
- population size or analysis window size
- (if RHEA): 0 (if initialization random) or 2 (if initialization MCTS)
- (if RHEA): true (if using shift buffer and MC rollouts at the end of individual evaluation) or false (if not used)
- game index (0-101)
- level index (0-4)
- instance index (0-19)

each row in a file = one game tick (each value sepated through space):
- Convergence
- Entropy actions explored
- 6 values: percentage each action was explored at any point during the search (first action of an individual in RHEA / action at the top of tree in MCTS), -1 if not at all
- Percentage of individuals recommending the final action chosen 
- Entropy actions recommended (in the final population for RHEA / final analysis window for MCTS)
- 6 values: percentage each action was recommended in the final pop for RHEA / final analysis window for MCTS
- Stat summary all fitness: MIN MAX MEAN Q1 MEDIAN Q3 SD STDERR ENTROPY
- 6 values: average fitness per action
- Number of times the algorithm saw wins during the search (using the Forward Model)
- 6 values: number of times the algorithm saw wins separated by the first action that led to the win
- Number of times the algorithm saw losses during the search (using the Forward Model)
- 6 values: number of times the algorithm saw losses separated by the first action that led to the loss

<hr>

**res/rhea-journal/actfiles**

file names (each value sepated through underscore _): 
- "act"
- "true" (if algorithm is RHEA) or "false" (if algorithm is MCTS)
- rollout length
- population size or analysis window size
- (if RHEA): 0 (if initialization random) or 2 (if initialization MCTS)
- (if RHEA): true (if using shift buffer and MC rollouts at the end of individual evaluation) or false (if not used)
- game index (0-101)
- level index (0-4)
- instance index (0-19)

each row in a file = one game tick (each value sepated through space):
- action played
- game score
(+ last row in the file: <random seed> <1 if win, 0 if loss> <game score> <game ticks>)

<hr>

* An "analysis window" is used for MCTS as a comparison to RHEA populations: if window size is L, then the first L iterations of MCTS will count in the 1st analysis window, the next L iterations will make the 2nd analysis window and so on.