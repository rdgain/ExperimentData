__author__ = 'dperez'

import matplotlib
matplotlib.use("Agg")
import glob
import numpy as np
import pylab
from config import *
import os
from scipy.stats import linregress
import math


def calc_results(this_result_dirs):

    train_games = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 19, 21, 23, 25, 27, 31, 34, 35, 38, 40, 41, 43, 45, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
               74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    test_games = [0, 11, 13, 14, 18, 20, 22, 24, 26, 28, 29, 30, 32, 33, 36, 37, 39, 42, 44, 46]

    alg = 0
    low = 0
    prev = 25
    high = 50
    resData = open("data/train/resultdatalevrep" + str(low) + "-" + str(high) + ".txt", 'w')

    for directory in this_result_dirs:

        victories = []
        scores = []
        times = []

        conv = []
        fitness = []
        expG = []
        expFM = []
        expTot = []
        countWin = []
        countLoss = []

        for game in range(NUM_GAMES):
            # if game not in train_games:
            #     continue

            victories_game = []
            scores_game = []
            times_game = []

            conv_game = []
            fitness_game = []
            expG_game = []
            expFM_game = []
            expTot_game = []
            countWin_game = []
            countLoss_game = []

            # print directory + "/results_" + str(game) + "_*.txt"
            # fileprefix = directory
            # if directory[0] == 'T':
            #     fileprefix = 'results_24_20'
            #     if directory[1] == 'M':
            #         fileprefix = 'results'

            dir = "rhea-journal/results/" + directory + "/" + directory + "_" + str(GAME_NAMES[game]) + ".txt"
            files = glob.glob(dir)
            if len(files) == 0 or os.stat(files[0]).st_size == 0:
                print "Missing or empty file: ", dir
                continue

            file = files[0]
            results = pylab.loadtxt(file, comments='*', delimiter=' ', usecols=range(1,15))
            resultsGame = results[1::2]
            resultsEvo = results[::2]

            l = 0
            numLines = 100
            rep = INSTANCES
            # rep = 1
            # for l in range(5):
            #     for r in range(INSTANCES):
            # start = l*rep + r
            start = 0
            # numLines = start+1

            victoriesAll = resultsGame[:,0][start:numLines]
            scoresAll = resultsGame[:,1][start:numLines]
            convAll = resultsEvo[:,0][start:numLines]
            eFAll = resultsEvo[:,9][start:numLines]
            expGAll = resultsEvo[:,10][start:numLines]
            expFMAll = resultsEvo[:,11][start:numLines]
            countWinAll = resultsEvo[:,12][start:numLines]
            countLossAll = resultsEvo[:,13][start:numLines]
            expTotAll = [expGAll[k]/expFMAll[k] for k in range(len(expGAll))]
            seeLW = [1 if (countLossAll[k] != -1 and victoriesAll[k] == 1) else 0 for k in range(len(countLossAll))]
            seeWL = [1 if (countWinAll[k] != -1 and victoriesAll[k] == 0) else 0 for k in range(len(countWinAll))]

            # Use 2000-times as time in all games where the game was lost.
            timesAllRaw = resultsGame[:,2][start:numLines]
            timesAll = [timesAllRaw[i] if victoriesAll[i] == 1.0 else (2000 - timesAllRaw[i]) for i in range(len(timesAllRaw))]

            # splitfile = file.split('/')[-1].split('.')[0].split('_')
            # newfile = ""
            # for f in range(1,len(splitfile)):
            #     newfile += splitfile[f] + "_"

            # actfile = "rhea-journal/actfiles/act_" + file.split('/')[-1].split('.')[0] + "_" + str(l) + "_" + str(r) + ".log"
            # if "true_" in file:
            #     evofile = "rhea-journal/evofiles/evo_" + newfile + str(l) + "_" + str(r) + ".log"
            # else:
            #     evofile = "rhea-journal/evofiles/mcts_" + newfile + str(l) + "_" + str(r) + ".log"
            #
            # resEvoFile = pylab.loadtxt(evofile, comments='*', delimiter=' ', usecols=range(45))
            # resActFile = pylab.loadtxt(actfile, comments='*', delimiter=' ', usecols=1)
            # resActFile = resActFile[:-1]
            #
            # if len(resActFile) != len(resEvoFile):  # this should not happen
            #     # print actfile
            #     continue

            # if len(resEvoFile) < prev:
            #     continue
            #
            # if len(resEvoFile) > high:
            #     resEvoFile = resEvoFile[low:high][:]
            #     resActFile = resActFile[low:high][:]

            # low_bound = int(low * len(resEvoFile) / 100.0)
            # high_bound = max(low_bound+1, int(math.ceil(high * len(resEvoFile) * 1.0 / 100)))
            # resEvoFile = resEvoFile[low_bound:high_bound][:]
            # resActFile = resActFile[low_bound:high_bound][:]
            #
            # # add from actfiles:
            # # - number of positive/negative scoring events
            # countPosScoreEv = 0
            # countNegScoreEv = 0
            # if len(resActFile) < 1:
            #     # print actfile
            #     print resActFile
            # lastScore = resActFile[0]
            # if len(resActFile) > 1:
            #     for li in range(1, len(resActFile)):
            #         line = resActFile[li]
            #         if line > lastScore:
            #             countPosScoreEv += 1
            #         if line < lastScore:
            #             countNegScoreEv += 1
            #         lastScore = line
            # currentScore = lastScore
            #
            # # add from evofiles
            # # - entropy acts recommended
            # # - entropy acts explored
            # # - entropy fitness per act
            # # - entropy win/loss count per act
            # # - win count slope
            # # - loss count slope
            # # - avg best fitness slope
            #
            # eWinpAct = 0
            # eLosspAct = 0
            # slopeWin = 0
            # slopeLoss = 0
            # slopeFit = 0
            # convAvg = 0
            #
            # try:
            #     eActRec = np.average(resEvoFile[:,9][:])
            #     eActExp = np.average(resEvoFile[:,1][:])
            #     eFitpAct = np.average(resEvoFile[:,24][:])
            #     winpAct = resEvoFile[:,32:37][:]
            #     losspAct = resEvoFile[:,39:44][:]
            #     convAvg = np.average(resEvoFile[:,0][:])
            #
            #     slopeWin, interceptWin, rvalueWin, pvalueWin, stderrWin = linregress(range(len(resEvoFile)),
            #                                                                          resEvoFile[:, 31][:])
            #     slopeLoss, interceptloss, rvalueLoss, pvalueLoss, stderrLoss = linregress(
            #         range(len(resEvoFile)), resEvoFile[:, 38][:])
            #     slopeFit, interceptFit, rvalueFit, pvalueFit, stderrFit = linregress(range(len(resEvoFile)),
            #                                                                          resEvoFile[:, 17][:])
            #
            #     eWin = []
            #     for row in winpAct:
            #         sh = 0
            #         if row.sum() != 0:
            #             pA = row / row.sum()
            #             for prob in pA:
            #                 if prob != 0:
            #                     sh -= np.sum(prob * np.log2(prob))
            #             eWin.append(sh)
            #     if len(eWin) > 0:
            #         eWinpAct = np.average(eWin)
            #
            #     eLoss = []
            #     for row in losspAct:
            #         sh = 0
            #         if row.sum() != 0:
            #             pA = row / row.sum()
            #             for prob in pA:
            #                 if prob != 0:
            #                     sh -= np.sum(prob * np.log2(prob))
            #         eLoss.append(sh)
            #     if len(eLoss) > 0:
            #         eLosspAct = np.average(eLoss)
            #
            # except:  # because some files have only 1 line..
            #     eActRec = resEvoFile[9]
            #     eActExp = resEvoFile[1]
            #     eFitpAct = resEvoFile[24]
            #     convAvg = resEvoFile[0]
            #
            #     winpAct = resEvoFile[32:37]
            #     losspAct = resEvoFile[39:44]
            #
            #     if winpAct.sum() != 0:
            #         pA = winpAct / winpAct.sum()
            #         for prob in pA:
            #             if prob != 0:
            #                 eWinpAct -= np.sum(prob * np.log2(prob))
            #     if losspAct.sum() != 0:
            #         pB = losspAct / losspAct.sum()
            #         for prob in pB:
            #             if prob != 0:
            #                 eLosspAct -= np.sum(prob * np.log2(prob))
            #
            # # save features to file
            # # stringToSave = str(game) + " " + str(alg) + " " + str(l) + " " + str(r) + " " + str(scoresAll[0]) + \
            # #                " " + str(timesAll[0]) + " " + str(convAll[0]) + " " + str(eFAll[0]) + \
            # #                " " + str(expGAll[0]) + " " + str(expFMAll[0]) + \
            # #                " " + str(countPosScoreEv) + " " + str(countNegScoreEv) + " " + str(slopeWin) + \
            # #                " " + str(slopeLoss) + " " + str(slopeFit) + " " + str(eActRec) + \
            # #                " " + str(eActExp) + " " + str(eFitpAct) + " " + str(eWinpAct) + \
            # #                " " + str(eLosspAct) + " " + str(victoriesAll[0]) + "\n"
            #
            # stringToSave = str(game) + " " + str(alg) + " " + str(l) + " " + str(r) + " " + str(currentScore) + \
            #                " " + str(convAvg) + " " + str(expGAll[0]) + " " + str(expFMAll[0]) + \
            #                " " + str(countPosScoreEv) + " " + str(countNegScoreEv) + " " + str(slopeWin) + \
            #                " " + str(slopeLoss) + " " + str(slopeFit) + " " + str(eActRec) + \
            #                " " + str(eActExp) + " " + str(eFitpAct) + " " + str(eWinpAct) + \
            #                " " + str(eLosspAct) + " " + str(victoriesAll[0]) + "\n"
            # resData.write(stringToSave)

            # get game features to add (same in all instances):
            # - number of NPCs
            # - number of portals
            # - number of resources

            ### from replay files:
            # - how many different obj types
            # - distance to closest obj of each type (proximity of N or more is reduced to N)

            all_victories_avg[game].append(np.average(victoriesAll))
            all_victories_stErr[game].append(np.std(victoriesAll) / np.sqrt(len(victoriesAll)))
            all_scores_avg[game].append(np.average(scoresAll))
            all_scores_stErr[game].append(np.std(scoresAll) / np.sqrt(len(scoresAll)))
            all_timesteps_avg[game].append(np.average(timesAll))
            all_timesteps_stErr[game].append(np.std(timesAll) / np.sqrt(len(timesAll)))

            all_conv_avg[game].append(np.average(convAll))
            all_conv_stErr[game].append(np.std(convAll) / np.sqrt(len(convAll)))
            all_fitness_avg[game].append(np.average(eFAll))
            all_fitness_stErr[game].append(np.std(eFAll) / np.sqrt(len(eFAll)))
            all_expG_avg[game].append(np.average(expGAll))
            all_expG_stErr[game].append(np.std(expGAll) / np.sqrt(len(expGAll)))
            all_expFM_avg[game].append(np.average(expFMAll))
            all_expFM_stErr[game].append(np.std(expFMAll) / np.sqrt(len(expFMAll)))
            all_expTot_avg[game].append(np.average(expTotAll))
            all_expTot_stErr[game].append(np.std(expTotAll) / np.sqrt(len(expTotAll)))
            all_countWin_avg[game].append(np.average(countWinAll))
            all_countWin_stErr[game].append(np.std(countWinAll) / np.sqrt(len(countWinAll)))
            all_countLoss_avg[game].append(np.average(countLossAll))
            all_countLoss_stErr[game].append(np.std(countLossAll) / np.sqrt(len(countLossAll)))
            all_seeLW[game] = seeLW
            all_seeWL[game] = seeWL

            victories_game.extend(victoriesAll)
            scores_game.extend(scoresAll)
            times_game.extend(timesAll)

            conv_game.extend(convAll)
            fitness_game.extend(eFAll)
            expG_game.extend(expGAll)
            expFM_game.extend(expFMAll)
            expTot_game.extend(expTotAll)
            countWin_game.extend(countWinAll)
            countLoss_game.extend(countLossAll)

            games_repetitions[game] = np.min([games_repetitions[game], len(victoriesAll)])

            victories.append(victories_game)
            scores.append(scores_game)
            times.append(times_game)

            conv.append(conv_game)
            fitness.append(fitness_game)
            expG.append(expG_game)
            expFM.append(expFM_game)
            expTot.append(expTot_game)
            countWin.append(countWin_game)
            countLoss.append(countLoss_game)

            # save this info to file:
            # G Alg % pts time conv e_R e_E f_slope e_F win_slope loss_slope e_W e_L %_expG/%_expFM avg_SeeLW avg_SeeWL

            # first just this, only results files needed:
            # G Alg % pts time conv e_F %_expG/%_expFM avg_SeeLW avg_SeeWL

            # stringToSave = str(game) + " " + str(alg) + " " + str(np.average(victoriesAll)) + " " + \
            #                str(np.average(scoresAll)) + " " + str(np.average(timesAll)) + " " + \
            #                str(np.average(convAll)) + " " + str(np.average(eFAll)) + " " + str(np.average(expTot)) + \
            #                " " + str(np.average(seeLW)) + " " + str(np.average(seeWL)) + "\n"
            # resData.write(stringToSave)

        overallVictories[alg].extend(victories)
        overallScores[alg].extend(scores)
        overallTimes[alg].extend(times)

        overallConv[alg].extend(conv)
        overallFitness[alg].extend(fitness)
        overallExpG[alg].extend(expG)
        overallExpFM[alg].extend(expFM)
        overallExpTot[alg].extend(expTot)
        overallCountWin[alg].extend(countWin)
        overallCountLoss[alg].extend(countLoss)

        v = np.hstack(victories)
        all_victories[alg] = [v]

        # perc_vict = np.average(v)
        # stdErr_vict = np.std(v) / np.sqrt(len(v))
        # print ALG_NAMES[alg], "Perc. victories: %.2f (%.2f), n=%.2f " % (perc_vict*100, stdErr_vict*100, len(v))

        alg += 1

    resData.close()



def calc_results_game(game_idx, this_result_dirs):



    NUM_GONFIGS = 6
    all_scores_avg = [[] for i in range(NUM_GONFIGS)]
    all_scores_stErr = [[] for i in range(NUM_GONFIGS)]

    all_victories_avg = [[] for i in range(NUM_GONFIGS)]
    all_victories_stErr = [[] for i in range(NUM_GONFIGS)]

    all_timesteps_avg = [[] for i in range(NUM_GONFIGS)]
    all_timesteps_stErr = [[] for i in range(NUM_GONFIGS)]

    games_repetitions = [REPS for i in range(NUM_GONFIGS)]
    config = 0


    for directory in this_result_dirs:

        victories = []
        scores = []
        times = []

        victories_game = []
        scores_game = []
        times_game = []
        alg = 0

        for dir in directory:

            # print dir + "/results_" + str(game_idx) + "_*.txt"

            files = glob.glob(dir + "/results_" + str(game_idx) + "_*.txt")
            if len(files) == 0:
                continue

            file = files[0]
            resultsGame = pylab.loadtxt(file, comments='*', delimiter=' ')
            # print "Reading", file

            numLines = min(REPS, resultsGame.shape[0])

            victoriesAll = resultsGame[:,0][0:numLines]
            scoresAll = resultsGame[:,1][0:numLines]

            # Use 2000-times as time in all games where the game was lost.
            timesAllRaw = resultsGame[:,2][0:numLines]
            timesAll = [ timesAllRaw[i] if victoriesAll[i] == 1.0 else (2000 - timesAllRaw[i]) for i in range(len(timesAllRaw))]

            all_victories_avg[alg].append(np.average(victoriesAll))
            all_victories_stErr[alg].append(np.std(victoriesAll) / np.sqrt(len(victoriesAll)))
            all_scores_avg[alg].append(np.average(scoresAll))
            all_scores_stErr[alg].append(np.std(scoresAll) / np.sqrt(len(scoresAll)))
            all_timesteps_avg[alg].append(np.average(timesAll))
            all_timesteps_stErr[alg].append(np.std(timesAll) / np.sqrt(len(timesAll)))

            # victories_game.extend(victoriesAll)
            # scores_game.extend(scoresAll)
            # times_game.extend(timesAll)

            games_repetitions[alg] = np.min([games_repetitions[alg], len(victoriesAll)])

            victories.append(victoriesAll)
            scores.append(scoresAll)
            times.append(timesAll)

            alg += 1



        overallVictories[config].extend(victories)
        overallScores[config].extend(scores)
        overallTimes[config].extend(times)

        v = np.hstack(victories)
        all_victories[config] = [v]

        config += 1


def compute_table_victories(data, data_stdErr, alg_names):

    dim1 = []
    dim2 = []

    data_idxs = dict()

    i = 0
    for alg in alg_names:
        dims = str.split(alg, '-')

        if not dim1.__contains__(dims[0]):
            dim1.append(dims[0])
        if not dim2.__contains__(dims[1]):
            dim2.append(dims[1])

        data_idxs[alg] = i
        i+=1


    table = [[0 for _ in range(len(dim2))] for _ in range(len(dim1))]
    table_std = [[0 for _ in range(len(dim2))] for _ in range(len(dim1))]


    for d1 in range(len(dim1)):
        for d2 in range(len(dim2)):
            alg_name = str(dim1[d1]) + '-' + str(dim2[d2])

            #print alg_name

            if data_idxs.has_key(alg_name):
                data_i = data_idxs[alg_name]
                table[d1][d2] = data[data_i]
                table_std[d1][d2] = data_stdErr[data_i]
            else:
                table[d1][d2] = '-'
                table_std[d1][d2] = '-'



    return dim1, dim2, table, table_std



def compute_table_victories_budget(data, data_stdErr, alg_names):

    dim1 = alg_names
    dim2 = ['24-20']

    data_idxs = dict()


    i = 0
    for alg in alg_names:
        data_idxs[alg] = i
        i+=1


    table = [[0 for _ in range(len(dim2))] for _ in range(len(dim1))]
    table_std = [[0 for _ in range(len(dim2))] for _ in range(len(dim1))]


    for d1 in range(len(alg_names)):
        alg_name = alg_names[d1]

        #print alg_name

        if data_idxs.has_key(alg_name):
            data_i = data_idxs[alg_name]
            table[d1][0] = data[data_i]
            table_std[d1][0] = data_stdErr[data_i]
        else:
            table[d1][0] = '-'
            table_std[d1][0] = '-'



    return dim1, dim2, table, table_std

