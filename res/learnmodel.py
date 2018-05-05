from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
np.set_printoptions(threshold=np.inf)
from graphs import *
from config import *
import random
from sklearn.model_selection import validation_curve
import datetime


file1 = "data/train/resultdatalevrep0-30.txt"
file2 = "data/train/resultdatalevrep0-30.txt"
# features = range(4, 20)
# file1 = "resultdatalevrep30-70.txt"  # training file
# file2 = "resultdatalevrep100-200.txt"  # test file
features = range(4, 18)
# features.remove(9)
# features.remove(8)
features.remove(7)
features.remove(6)
# features.remove(5)
print features

# features.remove(4)

print("Start time: " + str(datetime.datetime.now()))

metadata1 = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=range(4))
data1 = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=features)
classes1 = pylab.loadtxt(file1, comments='*', delimiter=' ', usecols=[18])

metadata2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=range(4))
data2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=features)
classes2 = pylab.loadtxt(file2, comments='*', delimiter=' ', usecols=[18])

# gamesbinary = pylab.loadtxt("gamedata/sp_games_binary.csv", comments='*', delimiter=',', usecols=0)
# gamesdiscr = pylab.loadtxt("gamedata/sp_games_discr.csv", comments='*', delimiter=',', usecols=0)
# gamesnonpc = pylab.loadtxt("gamedata/sp_games_nonpc.csv", comments='*', delimiter=',', usecols=0)
# gamesnpctype = pylab.loadtxt("gamedata/sp_games_npc_+type.csv", comments='*', delimiter=',', usecols=0)
# gamesnpce = pylab.loadtxt("gamedata/sp_games_npc_e.csv", comments='*', delimiter=',', usecols=0)
# gamesnpcf = pylab.loadtxt("gamedata/sp_games_npc_f.csv", comments='*', delimiter=',', usecols=0)
# gamesres = pylab.loadtxt("gamedata/sp_games_resources.csv", comments='*', delimiter=',', usecols=0)

# filtereddata = []
# for row in data:
#     # if row[0] in nowins:  # filter data by most wins games
#     #     continue
#     if row[0] in gamesbinary or row[0] in gamesdiscr:  # filter data by non-continuous rewards games
#         filtereddata.append(row)
# data = filtereddata

# samples = len(data)
# training_samples = 8*samples/10

train_games = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 19, 21, 23, 25, 27, 31, 34, 35, 38, 40, 41, 43, 45, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
               74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
test_games = [0, 11, 13, 14, 18, 20, 22, 24, 26, 28, 29, 30, 32, 33, 36, 37, 39, 42, 44, 46]


# games_win = [0] * 100
# for i in range(len(data)):  # get win count for all games
#     games_win[int(metadata[i][0])] += classes[i]
# games_win = sorted(games_win)
# train_gamesp = []
# test_gamesp = []
# alpha = 10
# for i in range(len(games_win)):
#     g = games_win[i]
#
#     if len(test_games) >= 20:
#         train_games.append(i)
#         train_gamesp.append(g)
#         continue
#     if len(train_games) >= 80:
#         test_games.append(i)
#         test_gamesp.append(g)
#         continue
#
#     if sum(train_gamesp) - sum(test_gamesp) > alpha:
#         test_games.append(i)
#         test_gamesp.append(g)
#         continue
#     if sum(train_gamesp) - sum(test_gamesp) < -alpha:
#         train_games.append(i)
#         train_gamesp.append(g)
#         continue
#
#     if random.uniform(0,1) < 0.5:
#         test_games.append(i)
#         test_gamesp.append(g)
#         continue
#     else:
#         train_games.append(i)
#         train_gamesp.append(g)
#         continue
#
# print len(train_games), train_games
# print len(test_games), test_games

X = []
y = []
X_test = []
y_test = []

for i in range(len(data1)):
    if metadata1[i][0] in train_games:
        X.append(data1[i])
        y.append(classes1[i])
        continue

for i in range(len(data2)):
    if metadata2[i][0] in test_games:
        X_test.append(data2[i])
        y_test.append(classes2[i])

print("Processed feature files: " + str(datetime.datetime.now()))

# X = data[0:training_samples]
# y = classes[0:training_samples]

perc_win = 0
for i in range(len(X)):
    if y[i] == 1:
        perc_win += 1
# print(perc_win * 1.0 / len(X))

# X_test = data[training_samples:samples]
# y_test = classes[training_samples:samples]

perc_win = 0
for i in range(len(X_test)):
    if y_test[i] == 1:
        perc_win+=1
# print(perc_win * 1.0 / len(X_test))

# print(np.shape(X), np.shape(y))
# print(np.shape(X_test), np.shape(y_test))

# names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "Dummy", "Win", "Lose"]
names = ["AdaBoost", "Win", "Lose"]

# classifiers = [
#     KNeighborsClassifier(3),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     DummyClassifier()]
classifiers = [AdaBoostClassifier()]

for c in range(len(classifiers)):
    print("Fitting classifier " + names[c] + ": " + str(datetime.datetime.now()))
    classifiers[c].fit(X, y)
    print("Finished fitting classifier, starting cross validation: " + str(datetime.datetime.now()))
    scores = cross_val_score(classifiers[c], X, y, cv=10)
    print("Finished cross validation: " + str(datetime.datetime.now()))
    print names[c] + " " + "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    # print classifiers[c].feature_importances_

    # param_range = range(20, 100, 10)
    # train_scores, test_scores = validation_curve(
    #     classifiers[c], X, y, param_name="n_estimators", param_range=param_range,
    #     cv=10, scoring="accuracy", n_jobs=1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    #
    # plt.title("Validation Curve with AdaBoost")
    # plt.xlabel("n_estimators")
    # plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    # lw = 2
    # plt.semilogx(param_range, train_scores_mean, label="Training score",
    #              color="darkorange", lw=lw)
    # plt.fill_between(param_range, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.2,
    #                  color="darkorange", lw=lw)
    # plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    #              color="navy", lw=lw)
    # plt.fill_between(param_range, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.2,
    #                  color="navy", lw=lw)
    # plt.legend(loc="best")
    # plt.show()

    # if c == 1:
    #     print classifiers[c].feature_importances_

classifiers.append(None)
y_pred = [[] for _ in range(len(classifiers))]
creport = []
loss_pred = [0.0] * len(y_test)
win_pred = [1.0] * len(y_test)

print "-------"
for c in range(len(classifiers)):
    if classifiers[c] is not None:
        clf = classifiers[c]
        print("Starting prediction with classifier " + names[c] + ": " + str(datetime.datetime.now()))
        for i in range(len(y_test)):
            prediction = clf.predict(X_test[i].reshape(1, -1))
            y_pred[c].extend(prediction)
        print("Finished prediction: " + str(datetime.datetime.now()))
    else:
        for i in range(len(y_test)):
            if X_test[i][2] > X_test[i][3]:
                prediction = 1
            else:
                prediction = 0
            y_pred[c].append(prediction)
    creport.append(classification_report(y_test, y_pred[c]))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred[c]).ravel()
    prec = tp * 1.0/(tp + fp)
    rec = tp * 1.0/(tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    acc = (tp + tn) * 1.0 / (tp + tn + fp + fn)
    print names[c], "%.2f" % acc, "%.2f" % prec, "%.2f" % rec, "%.2f" % f1

# tnl, fpl, fnl, tpl = confusion_matrix(y_test, loss_pred).ravel()
# tnw, fpw, fnw, tpw = confusion_matrix(y_test, win_pred).ravel()
creport.append(classification_report(y_test, win_pred))
creport.append(classification_report(y_test, loss_pred))

for r in range(len(creport)):
    report = creport[r]
    print names[r]
    print report
    # plot_classification_report(report)
    # plt.savefig('report_' + names[r] + '.png', dpi=200, format='png', bbox_inches='tight')
    # plt.close()


