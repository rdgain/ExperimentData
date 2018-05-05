from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import pickle
import pylab
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular
import random
from config_classifier import *


global model_names, model_names_p, model_colors, models_test

if not TRAIN:  # only use some of the models
    model_names = [model_names[i] for i in range(len(model_names)) if i in models_test]
    last = model_names_p[-1]
    lastc = model_colors[-1]
    model_names_p = [model_names_p[i] for i in range(len(model_names_p)) if i in models_test]
    model_colors = [model_colors[i] for i in range(len(model_colors)) if i in models_test]
    if INCLUDE_RULE_MODEL:
        model_names_p.append(last)
        model_colors.append(lastc)


# Train all 3 models for early, mid, late game
if TRAIN:  # Train models
    for f in range(len(file_train)):
        file1 = "data/train/resultdatalevrep" + file_train[f] + ".txt"

        games = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=[game_column])
        algs = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=[alg_column])
        data1 = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=features)
        classes1 = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=[target_class_column])

        X = []
        y = []

        for i in range(len(data1)):
            if games[i] in train_games and algs[i] in train_algs:
                X.append(data1[i])
                y.append(classes1[i])
                continue

        # explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X), feature_names=feature_names,
        #                                                    class_names=class_names)
        # explainers.append(explainer)

        clf = AdaBoostClassifier()
        clf.fit(X, y)
        # print model_names[f] + " " + str(clf.feature_importances_)
        # pickle.dump(clf, open("models/" + model_names[f] + ".pkl", 'wb'))
        pickle.dump(clf, open("models-newalgs/mcts_" + model_names[f] + ".pkl", 'wb'))

else:  # Test models
    for name in model_names:  # get models
        # loaded_model = pickle.load(open("models/" + name + ".pkl", 'rb'))
        loaded_model = pickle.load(open("models-newalgs/mcts_" + name + ".pkl", 'rb'))
        models.append(loaded_model)

    if INCLUDE_RULE_MODEL:
        models.append(None)

    y_pred_perc = [[[] for _ in range(len(models))] for _ in range(no_games)]
    y_pred_acc = [[[] for _ in range(len(models))] for _ in range(no_games)]
    y_pred_err = [[[] for _ in range(len(models))] for _ in range(no_games)]
    win_avg_game = [0] * no_games
    win_serr_game = [0] * no_games

    # predict stuff
    for t in test_cases:  # for each game tick
        print "Beginning test case " + str(t)

        file2 = "data/test/resultdatalevrep0-" + str(t) + ".txt"
        metadata2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[game_column])
        alg2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[alg_column])
        lvl2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[lvl_column])
        data2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=features)
        classes2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[target_class_column])

        X_test = [[] for _ in range(no_games)]
        y_test = [[] for _ in range(no_games)]
        y_pred = [[[] for _ in range(len(models))] for _ in range(no_games)]

        games = []
        algs = []
        lvls = []
        for i in range(len(data2)):  # get test data
            game = int(metadata2[i])
            alg = int(alg2[i])
            lvl = int(lvl2[i])
            if game in test_games and alg in test_algs:  # and game not in games:  # only get 1 instance per game
                X_test[game].append(data2[i])
                y_test[game].append(classes2[i])
                games.append(game)
                algs.append(alg)
                lvls.append(lvl)
        games = list(set(games))

        # record win percentage for each game
        if t == 100:
            for g in games:
                win_avg_game[g] = np.average(y_test[g])
                win_serr_game[g] = np.std(y_test[g])

        print "Got test data, preparing to predict..."

        for g in range(len(games)):  # for each game
            game = int(games[g])
            if game != 44:
                continue

            print "Predicting game " + str(game)

            for m in range(len(models)):  # for each model

                print "Predicting with model " + model_names_p[m] + " game " + str(game) + " tick " + str(t)

                clf = models[m]
                pred_list = []
                for i in range(len(y_test[game])):  # predict win by model m in game at game tick t
                    if INCLUDE_RULE_MODEL and len(models) > 1 and m != len(models) - 1 or not INCLUDE_RULE_MODEL:  # for all but last model
                        prediction = clf.predict(X_test[game][i].reshape(1, -1))

                        # explaining prediction, saves html file and pyplot figure
                        # sometimes gives odd exceptions which are ignored for now
                        # try:
                        #     exp = explainers[m].explain_instance(X_test[game][i], clf.predict_proba,
                        #                                          num_features=len(features), top_labels=1)
                        #     path = model_names_p[m] + '_' + str(game) + '_' + str(t) + '_' + str(i)
                        #     figure = exp.as_pyplot_figure()
                        #     plt.title("")
                        #     plt.ylabel("Feature")
                        #     plt.xlabel("Class recommendation")
                        #     plt.tight_layout()
                        #     figure.savefig(path + '.png')
                        #     exp.save_to_file(path + '.html')
                        #     print(exp.as_list())
                        # except:
                        #     pass

                    else:  # ------- simple rule-based prediction: if you're gaining score, you're winning
                        if X_test[game][i][2] > X_test[game][i][3]:
                            prediction = 1
                        else:
                            prediction = 0
                    # y_pred[game][m].append((1 if prediction == y_test[game][i] else 0))
                    y_pred[game][m].append(prediction)
                    pred_list.append(prediction)

                # print classification report for model x game x tick
                # print(classification_report(y_test[game], np.array(pred_list).flatten().tolist()))

        print "Predictions done, computing averages..."

        for g in range(len(games)):  # for each game
            game = int(games[g])
            if game != 44:
                continue
            for m in range(len(models)):  # for each model
                # record predict percentage for this game / game tick / model
                pred = y_pred[game][m]
                f1 = f1_score(y_test[game], pred)
                acc = accuracy_score(y_test[game], pred)
                y_pred_perc[game][m].append(f1)
                y_pred_acc[game][m].append(acc)
                y_pred_err[game][m].append(np.std(pred) / np.sqrt(len(pred)))

    print "All test cases finished, plotting..."

    # Save data to files
    for g in range(len(y_pred_perc)):
        if g in test_games:
            if g != 44:
                continue
            f = open("data/predicted/mcts_" + str(g) + ".txt", 'w')
            f1 = open("data/predicted/win_mcts_" + str(g) + ".txt", 'w')

            for m in range(len(y_pred_perc[g])):
                for value in y_pred_perc[g][m]:
                    f.write(str(value) + " ")
                f.write("\n")
                for value in y_pred_acc[g][m]:
                    f.write(str(value) + " ")
                f.write("\n")
                for value in y_pred_err[g][m]:
                    f.write(str(value) + " ")
                f.write("\n")

            f1.write(str(win_avg_game[g]) + " " + str(win_serr_game[g]) + "\n")

            f.close()
            f1.close()
