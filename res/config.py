__author__ = 'dperez'


mcts_in = True
rh_in = True
rs_in = True

all_result_dirs = []
ALG_NAMES = []
ALG_NAMES_LETTER = []
F1_RANKING_POINTS = []
alg_index = []

pop_size = [2, 10]
ind_length = [8, 14]

pop_size2 = [10, 90, 30]
ind_length2 = [90, 10, 30]

window_size = [2, 10, 10]
sim_depth = [8, 14, 10]

init = [0, 2]
buffer = ['false', 'true']
rhea = 'true'
mcts = 'false'

if rs_in:
    pop_size.extend(pop_size2)
    ind_length.extend(ind_length2)

CONFIG = "All"
HM_TITLE = "All"
HM_FILENAME = 'RHEA'
# if len(rollouts) > 1 or rollouts[0] != 0 :
#     HM_TITLE = "R = " + str(repeat[-1])
#     HM_FILENAME += "-rollouts"
#     if mcts_in :
#         HM_FILENAME += "-mcts"
#     HM_FILENAME += '-' + str(repeat[-1])
# else :
#     HM_TITLE = CONFIG


F1_RANKING_POINTS_BASE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]

idx = 0
counter = 65  # 'A'

if rh_in:
    for p in range(len(pop_size)):
        i = pop_size[p]
        j = ind_length[p]
        for s in range(len(init)):
            if j in ind_length2 and s != 0:
                continue
            for t in range(len(buffer)):
                if j in ind_length2 and t != 0:
                    continue

                n = init[s]
                m = buffer[t]

                d = rhea + "_" + str(j) + "_" + str(i) + "_" + str(n) + "_" + str(m)

                name0 = "EA"
                name1 = ""
                name2 = ""
                if m == 'true':
                    name1 += "-shift-roll"
                if n == 2:
                    name2 += "-mcts"

                fullname = name0 + name1 + name2
                if j in ind_length2:
                    fullname = "RS"
                if m == 'false' and n == 0:
                    fullname = "Vanilla"

                all_result_dirs.append(d)
                ALG_NAMES.append(str(i) + "-" + str(j) + "-" + fullname)
                ALG_NAMES_LETTER.append(chr(counter))
                counter += 1

                if counter == 91:
                    counter = 97  # Jump to 'a'

                if idx <+ 10:
                    F1_RANKING_POINTS.append(F1_RANKING_POINTS_BASE[idx])
                else:
                    F1_RANKING_POINTS.append(0)

                idx += 1

if mcts_in:
    for i in range(len(sim_depth)):
        ln = sim_depth[i]
        ws = window_size[i]
        d = mcts + "_" + str(ln) + "_" + str(ws)
        all_result_dirs.append(d)
        ALG_NAMES.append("MCTS-" + str(ws) + "-" + str(ln))
        ALG_NAMES_LETTER.append(chr(counter))
        counter += 1

        if counter == 91:
            counter = 97  # Jump to 'a'

        if idx <+ 10:
            F1_RANKING_POINTS.append(F1_RANKING_POINTS_BASE[idx])
        else:
            F1_RANKING_POINTS.append(0)

        idx += 1


for i in range(len(all_result_dirs)):  # all
    alg_index.append(i)

N_ALG = len(all_result_dirs)

GAME_NAMES = range(102)  # All games

# remove problematic games
GAME_NAMES.remove(73)
GAME_NAMES.remove(83)

nowins = [99,59,55,38,6,23,82,32,34,85,47,43,96,92,90,89,75,72,71,63,58,57,56,53,39,31,30,29,28,24,21,17,2,14,7,46,4,35,3,86,5,15,12,54,80,76,81,16,45,33]

# for i in nowins:
#     GAME_NAMES.remove(i)

# GAME_NAMES = ["aliens", "bait", "blacksmoke", "boloadventures", "boulderchase",              # 0-4
#                 "boulderdash", "brainman", "butterflies", "cakybaky", "camelRace",     # 5-9
#                 "catapults", "chase", "chipschallenge", "chopper", "cookmepasta",        # 10-14
#                 "crossfire", "defem", "defender", "digdug", "eggomania",           # 15-19
#                 "enemycitadel", "escape", "factorymanager", "firecaster",  "firestorms",   # 20-24
#                 "frogs", "gymkhana", "hungrybirds", "iceandfire", "infection",    # 25-29
#                 "intersection", "jaws", "labyrinth", "lasers", "lasers2",        # 30-34
#                 "lemmings", "missilecommand", "modality", "overload", "pacman",             # 35-39
#                 "painter", "plants", "plaqueattack", "portals", "raceBet2",         # 40-44
#                 "realportals", "realsokoban", "roguelike", "seaquest", "sheriff",      # 45-49
#                 "sokoban", "solarfox" ,"superman", "surround", "survivezombies", # 50-54
#                 "tercio", "thecitadel", "waitforbreakfast", "watergame", "whackamole", # 55-59
#                 "zelda", "zenpuzzle", # 60-61
#                 "angelsdemons", "assemblyline", "avoidgeorge", "bomber", "chainreaction", # 62-26
#                 "clusters", "colourescape", "cops", "dungeon", "fireman", # 67-71
#                 "freeway", "islands", "labyrinthdual", "racebet",  "rivers",   # 72-76
#                 "run", "shipwreck", "thesnowman", "waves", "witnessprotection"]


NUM_GAMES = len(GAME_NAMES)
NUM_LEVELS = 5
REPS = 20 * NUM_LEVELS
INSTANCES = 20
SIGNIFICANCE_P_VALUE = 0.05

G = NUM_GAMES + 1

all_victories = [[] for i in range(N_ALG)]
overallVictories = [[] for i in range(N_ALG)]
overallScores = [[] for i in range(N_ALG)]
overallTimes = [[] for i in range(N_ALG)]

overallConv = [[] for i in range(N_ALG)]
overallFitness = [[] for i in range(N_ALG)]
overallExpG = [[] for i in range(N_ALG)]
overallExpFM = [[] for i in range(N_ALG)]
overallExpTot = [[] for i in range(N_ALG)]
overallCountWin = [[] for i in range(N_ALG)]
overallCountLoss = [[] for i in range(N_ALG)]

# game stats

games_repetitions = [REPS for i in range(NUM_GAMES)]

all_scores_avg = [[] for i in range(NUM_GAMES)]
all_scores_stErr = [[] for i in range(NUM_GAMES)]

all_victories_avg = [[] for i in range(NUM_GAMES)]
all_victories_stErr = [[] for i in range(NUM_GAMES)]

all_timesteps_avg = [[] for i in range(NUM_GAMES)]
all_timesteps_stErr = [[] for i in range(NUM_GAMES)]

all_conv_avg = [[] for i in range(NUM_GAMES)]
all_conv_stErr = [[] for i in range(NUM_GAMES)]

all_fitness_avg = [[] for i in range(NUM_GAMES)]
all_fitness_stErr = [[] for i in range(NUM_GAMES)]

all_expG_avg = [[] for i in range(NUM_GAMES)]
all_expG_stErr = [[] for i in range(NUM_GAMES)]

all_expFM_avg = [[] for i in range(NUM_GAMES)]
all_expFM_stErr = [[] for i in range(NUM_GAMES)]

all_expTot_avg = [[] for i in range(NUM_GAMES)]
all_expTot_stErr = [[] for i in range(NUM_GAMES)]

all_countWin_avg = [[] for i in range(NUM_GAMES)]
all_countWin_stErr = [[] for i in range(NUM_GAMES)]

all_countLoss_avg = [[] for i in range(NUM_GAMES)]
all_countLoss_stErr = [[] for i in range(NUM_GAMES)]

all_seeLW = [0 for i in range(NUM_GAMES)]
all_seeWL = [0 for i in range(NUM_GAMES)]