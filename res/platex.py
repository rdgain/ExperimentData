from __future__ import print_function

"""
make table average accuracy and f1 scores across all games for each model
need to test on train data files (but test games) 0-30, 0-70, 0-100 for each model
plus average of averages for overall
"""
from config_classifier import *
import pickle
import pylab
import numpy as np
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


for name in model_names:  # get models
    loaded_model = pickle.load(open("models/" + name + ".pkl", 'rb'))
    models.append(loaded_model)

if INCLUDE_RULE_MODEL:
    models.append(None)

test_files = ["0-15", "0-50", "0-85"]
phase_names = ["Phase", "Early", "Mid", "Late", "Avg / Total"]

y_pred_perc = [[np.nan for _ in range(len(models) + 1)] for _ in range(len(test_files) + 1)]
y_pred_err = [[np.nan for _ in range(len(models) + 1)] for _ in range(len(test_files) + 1)]

global target_class_column, features

for k in range(len(test_files)):
    f = test_files[k]

    file2 = "data/train/resultdatalevrep" + f + ".txt"

    if f == "":  # full feature file, need to adjust features
        target_class_column = 20
        features = range(instance_column + 1, target_class_column)
        features.remove(9)
        features.remove(8)
        features.remove(7)
        features.remove(5)

    # get data from file

    game2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[game_column])
    alg2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[alg_column])
    data2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=features)
    classes2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[target_class_column])

    X_test = []
    y_test = []

    for i in range(len(data2)):  # get test data
        game = int(game2[i])
        alg = int(alg2[i])
        if game in test_games and alg in test_algs:  # and game not in games:  # only get 1 instance per game
            X_test.append(data2[i])
            y_test.append(classes2[i])

    print("Got test data, preparing to predict...")

    for m in range(len(models)):
        print("Predicting with model " + model_names_p[m] + " in range " + (f if f != "" else "0-100"))
        clf = models[m]
        if clf is not None:
            prediction = clf.predict(X_test)
        else:
            prediction = [1 if X_test[i][2] > X_test[i][3] else 0 for i in range(len(X_test))]

        # y_pred_perc[k][m] = np.average(prediction)
        # y_pred_err[k][m] = np.std(prediction) / np.sqrt(len(prediction))

        f1 = f1_score(y_test, prediction)
        acc = accuracy_score(y_test, prediction)
        y_pred_perc[k][m] = f1
        y_pred_err[k][m] = acc

print("Printing table (model rows, phase columns) ...")

# add mean by row and column
a = np.array(y_pred_perc)
b = np.array(y_pred_err)
for phase in range(len(y_pred_perc)):
    a[phase][-1] = np.nanmean(a, axis=1)[phase]
    b[phase][-1] = np.nanmean(b, axis=1)[phase]

    for model in range(len(y_pred_perc[phase])):
        a[-1][model] = np.nanmean(a, axis=0)[model]
        b[-1][model] = np.nanmean(b, axis=0)[model]

c = y_pred_err
for p in range(len(y_pred_err)):
    for m in range(len(y_pred_err[p])):
        c[p][m] = "%0.2f (%0.2f)" % (a[p][m], b[p][m])

print(np.matrix(np.array(c).transpose()))

