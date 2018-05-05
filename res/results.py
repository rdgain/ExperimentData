__author__ = 'dperez'
import numpy as np
from scipy.stats import mannwhitneyu as nonparam
from numpy import std, mean, sqrt


def nonparamTest(data, i, j, game):

    if (np.average(data[i][game]) != np.average(data[j][game])):
       return nonparam(data[i][game], data[j][game])
    return 1.0, 1.0



"""
    Computes statistical tests (Mann Whit) between algorithms in all games, using averages of victories, scores and times.
    Returns three multi-dimensional arrays (v,s,t). For each game (first dimension), an array n_alg x n_alg array is
    provided, with the p-values in the upper diagonal between the different algorithms.
"""
def compute_statistical_tests(overallVictories, overallScores, overallTimes, num_games, result_dirs, alg_index):

    all_victories_test = []
    all_scores_test = []
    all_times_test = []
#    n_alg = len(result_dirs)
   
    n_alg = len(alg_index)

    for game in range(num_games):

        victories_test = [[0 for i in range(n_alg)] for j in range(n_alg)]
        scores_test = [[0 for i in range(n_alg)] for j in range(n_alg)]
        times_test = [[0 for i in range(n_alg)] for j in range(n_alg)]

        data_sc = []
        data_vic = []
        data_times = []

        #Take one game and compare all approaches.
        for approach in range(n_alg):

            #Normality test
            # g,p_normal = normality_test( overallVictories[approach][game])
            # normal = "not normal"
            # if(p_normal > 0.01):
            #     normal = "normal"
            # print 0,p_normal, normal

            data_sc.append(np.asarray(overallScores[alg_index[approach]][game]))
            data_vic.append(np.asarray(overallVictories[alg_index[approach]][game]))
            data_times.append(np.asarray(overallTimes[alg_index[approach]][game]))

            for approach2 in range(approach, n_alg):

                if(approach != approach2):

                    t, pVal = nonparamTest(overallVictories, alg_index[approach], alg_index[approach2], game)
                    victories_test[approach][approach2] = pVal

                    # Calculate effect size
                    if pVal < 0.05 :
                        # print pVal
                        x = overallVictories[alg_index[approach]][game]
                        y = overallVictories[alg_index[approach2]][game]
                        # cohens_d = (mean(x) - mean(y)) / (sqrt((std(x) ** 2 + std(y) ** 2) / 2))
                        # print("victories " + str(game) + " -- " + result_dirs[alg_index[approach]] + " -- " + result_dirs[alg_index[approach2]] + " -- " + str(pVal) + " -- " + str(cohens_d))

                    t, pVal = nonparamTest(overallScores, alg_index[approach], alg_index[approach2], game)
                    scores_test[approach][approach2] = pVal

                    if pVal < 0.05 :
                        #print pVal
                        x = overallScores[alg_index[approach]][game]
                        y = overallScores[alg_index[approach2]][game]
                        # cohens_d = (mean(x) - mean(y)) / (sqrt((std(x) ** 2 + std(y) ** 2) / 2))
                        # print("scores " + str(game) + " -- " + result_dirs[alg_index[approach]] + " -- " + result_dirs[alg_index[approach2]] + " -- " + str(pVal) + " -- " + str(cohens_d))


                    # t, pVal = nonparamTest(overallTimes, approach, approach2, game)
                    # times_test[approach][approach2] = pVal


        all_victories_test.append(victories_test)
        all_scores_test.append(scores_test)
        # all_times_test.append(times_test)

    return all_victories_test, all_scores_test, all_times_test



"""
    Builds a comprehensible table with all results, with this format:

    Game    Algorithm       Victories avg (StdErr) | Stat.Better | Scores avg (StdErr) | Stat.Better
    game    A: SampleMCTS 	0.98 (0.01)     [B, D] (=> means it is statistically better than B and D).
    game    B: MCTSWghts 	0.95 (0.01)     [D]
    game    C: MCTSMixed 	0.98 (0.01)     [B, D]
    game    D: ParetoMCTS 	0.91 (0.01)     []

    This version
"""
def build_results_table(all_victories_avg, all_victories_stErr, all_victories_test,
                 all_scores_avg, all_scores_stErr, all_scores_test,
                 all_times_avg, all_times_stErr, all_times_test,
                 sig_p_value, games_repetitions, game_names, n_games, alg_names, alg_names_letter, alg_index):

    n_alg = len(alg_index)

    num_games = n_games
    prettyTable = [[] for i in range(num_games * n_alg)]

    for game in range(num_games):

        sig_better_vic = [[] for i in range(n_alg)]
        sig_better_sc = [[] for i in range(n_alg)]
        sig_better_times = [[] for i in range(n_alg)]

        for approach in range(n_alg):
            prettyIdx = game*n_alg + approach
            prettyTable[prettyIdx] = ["" for i in range(6)] #Hardcoded number of elements are in the list we want to compose.

            # prettyTable[prettyIdx][0] = game_names[game] + " (" + str(games_repetitions[game]) + ")"
            prettyTable[prettyIdx][0] = game_names[game]
            prettyTable[prettyIdx][1] = alg_names_letter[alg_index[approach]] + ": " + alg_names[alg_index[approach]]
            prettyTable[prettyIdx][2] = "$%.2f$ $(%.2f)$" % (all_victories_avg[game][alg_index[approach]]*100, all_victories_stErr[game][alg_index[approach]] * 100)
            prettyTable[prettyIdx][4] = "$%.2f$ $(%.2f)$" % (all_scores_avg[game][alg_index[approach]], all_scores_stErr[game][alg_index[approach]])
            # prettyTable[prettyIdx][6] = "$%.2f$ $(%.2f)$" % (all_times_avg[game][approach], all_times_stErr[game][approach])

            for approach2 in range(approach, n_alg):
                if(approach != approach2):

##                    print alg_index[approach], alg_index[approach2]

                    # Victories
                    pVal = all_victories_test[game][approach][approach2]
                    if pVal < sig_p_value:
                        #print "victories", pVal, approach, approach2
                        if all_victories_avg[game][alg_index[approach]] > all_victories_avg[game][alg_index[approach2]]:
                            sig_better_vic[approach].append(alg_names_letter[alg_index[approach2]])
                        else:
                            sig_better_vic[approach2].append(alg_names_letter[alg_index[approach]])

                    # Scores
                    pVal = all_scores_test[game][approach][approach2]
                    if pVal < sig_p_value:
                        #print "scores", pVal, approach, approach2
                        if all_scores_avg[game][alg_index[approach]] > all_scores_avg[game][alg_index[approach2]]:
                            sig_better_sc[approach].append(alg_names_letter[alg_index[approach2]])
                        else:
                            sig_better_sc[approach2].append(alg_names_letter[alg_index[approach]])


                    # # Times
                    # pVal = all_times_test[game][approach][approach2]
                    # if pVal < sig_p_value:
                    #     #print "times", pVal, approach, approach2
                    #     if all_times_avg[game][approach] < all_times_avg[game][approach2]:
                    #         sig_better_times[approach].append(alg_names_letter[approach2])
                    #     else:
                    #         sig_better_times[approach2].append(alg_names_letter[approach])


            prettyTable[prettyIdx][3] = sig_better_vic[approach]
            prettyTable[prettyIdx][5] = sig_better_sc[approach]
            # prettyTable[prettyIdx][7] = sig_better_times[approach]

    
    return prettyTable
