from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from src.config import *


# Names of classifiers
names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "Dummy"]

# List of classifiers
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    DummyClassifier()]


"""
Function to train all classifiers on the training data
"""
def train(X, y):
    for c in range(len(classifiers)):
        classifiers[c].fit(X, y)  # fit classifier to data

        # use cross-validation during training and print accuracy
        scores = cross_val_score(classifiers[c], X, y, cv=10)
        print(names[c] + " " + "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # print Decision Tree feature importance
        if c == 1:
            print(classifiers[c].feature_importances_)
    print()


"""
Function to test all classifiers on the test data and compute accuracy
"""
def test(X_test, y_test):
    y_pred = [[] for _ in range(len(classifiers))]
    loss_pred = ['Fail'] * len(y_test)  # predictor which always returns 'Fail'
    win_pred = ['Win'] * len(y_test)  # predictor which always returns 'Win'

    for c in range(len(classifiers)):
        clf = classifiers[c]
        for i in range(len(y_test)):  # get all classifiers to predict result on test data
            prediction = clf.predict(np.array(X_test[i]).reshape(1, -1))
            y_pred[c].append(prediction)
        print(names[c])
        print(classification_report(y_test, np.array(y_pred[c]).flatten().tolist()))  # print classification report

    # Print Win and Fail classification reports
    print("Win")
    print(classification_report(y_test, win_pred))
    print("Fail")
    print(classification_report(y_test, loss_pred))
    print()


"""
Function to extract features from the data and save them to CSV files
"""
def extract_features():
    features = [[[] for _ in range(len(teamCS))] for _ in range(no_players)]
    features_all = [[] for _ in range(len(teamCS))]

    classes = [[] for _ in range(no_players)]
    classes_all = []

    for p in range(no_players):
        f = open("../data/features_" + players[p] + ".csv", 'w')
        for i in range(len(teamCS)):
            features[p][i].append(playerCS[p][i])
            features[p][i].append(playerGold[p][i])
            features[p][i].append(playerXP[p][i])
            features[p][i].append(playerDmg[p][i])
            classes[p].append(win[i])

            features_all[i].extend(features[p][i])
            if p == 0:
                classes_all.append(win[i])

            # write feature to CSV files
            f.write(str(playerCS[p][i]) + " ")
            f.write(str(playerGold[p][i]) + " ")
            f.write(str(playerXP[p][i]) + " ")
            f.write(str(playerDmg[p][i]) + " ")
            f.write(str(win[i]) + "\n")

        f.close()

    with open("../data/features_team.csv", 'w') as f:
        for i in range(len(teamCS)):
            for feature in features_all[i]:
                f.write(str(feature) + " ")
            f.write(str(win[i]) + "\n")

    return features, features_all, classes, classes_all


"""
Function to classify the data. Extracts the features, separates data into train and test and runs classifiers.
"""
def classify():
    # extract features of interest from the data and gather them all in data structures
    features, features_all, classes, classes_all = extract_features()

    # separate training and testing data sets

    samples = len(features[0])
    training_samples = int(0.8 * samples)

    X = [features[i][0:training_samples] for i in range(len(features))]
    y = [classes[i][0:training_samples] for i in range(len(features))]

    X_all = features_all[0:training_samples]
    y_all = classes_all[0:training_samples]

    X_test = [features[i][training_samples:samples] for i in range(len(features))]
    y_test = [classes[i][training_samples:samples] for i in range(len(features))]

    X_all_test = features_all[training_samples:samples]
    y_all_test = classes_all[training_samples:samples]

    # train and test classifiers

    for p in range(no_players):
        print(players[p])
        train(X[p], y[p])
        test(X_test[p], y_test[p])

    train(X_all, y_all)
    test(X_all_test, y_all_test)