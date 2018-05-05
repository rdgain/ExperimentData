TRAIN = False  # if false, testing.
INCLUDE_RULE_MODEL = True

train_games = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 19, 21, 23, 25, 27, 31, 34, 35, 38, 40, 41, 43, 45, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
               74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
test_games = [0, 11, 13, 14, 18, 20, 22, 24, 26, 28, 29, 30, 32, 33, 36, 37, 39, 42, 44, 46]
# test_games = [0]

target_class_column = 18
game_column = 0
alg_column = 1
lvl_column = 2
instance_column = 3
no_algs = 14
no_games = 100

features = [i for i in range(instance_column + 1, target_class_column) if i != 7 and i != 6]
# exp FM % exp game %

feature_names = [r'$\phi_1$', r'$\phi_2$', r'$\phi_3$',
                 r'$\phi_4$', r'$\phi_5$', r'$\phi_6$', r'$\phi_7$',
                 r'$\phi_8$', r'$\phi_9$', r'$\phi_{10}$', r'$\phi_{11}$', r'$\phi_{12}$']
class_names = [0, 1]

# training files and models
file_train = ["0-10", "0-20", "0-30", "0-50", "20-80", "30-70", "35-65", "50-100", "70-100", "80-100", "90-100"]
# file_train = ["0-30", "30-70", "70-100"]
model_names = ["early1", "early2", "early3", "half1", "mid1", "mid2", "mid3", "half2", "late1", "late2", "late3"]
model_names_p = ["Early game 1", "Early game 2", "Early game", "First half game", "Mid game 1",
                 "Mid game", "Mid game 3", "Second half game", "Late game", "Late game 2", "Late game 3",
                 "Rule based"]
model_colors = ["#3f2ac9", "#4a67c4", "#40b5e8", "black", "#edb138", "#e58c20", "#ef8143", "gray", "#87e23d", "#2bad38",
                "#1dc487", "red"]
model_style = ['-.', '-.', '-.', ':', '-', '-', '-', ':', '--', '--', '--', ':']
test_cases = range(100, 2001, 100)  # every 100 game ticks
models_test = [2, 5, 8]

# filter algs, use only 80% for train, and 20% new ones for test
# algs = range(14)
# no_algs = 11
# train_algs = [algs[i] for i in sorted(random.sample(xrange(len(algs)), no_algs))]
# print train_algs
# train_algs = range(no_algs)
# train_algs = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13]
train_algs = range(11)  # RHEA+RS only
# test_algs = range(no_algs)
# test_algs = [0, 5, 9]
test_algs = [11, 12, 13]  # MCTS only

explainers = []
models = []
