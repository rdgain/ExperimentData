from compute import *
from graphs import *
from latex import *
from results import *
from rankings import *
# from tkinter import *
import datetime

gInput = 0  # game input, can be a single number, a range as A-B or a sequence as A,B,C,D
exclGInput = 0  # games excluded, can be a single number, a range as A-B or a sequence as A,B,C,D
fInput = 0  # feature to plot
algInput = 0  # alg to plot, can be a single number, a range as A-B or a sequence as A,B,C,D

def results(dirs):
    # Compute the results to populate the arrays and structures of this file.
    calc_results(dirs)

def stat_tests(dirs, alg_index):
    # Compute the statistical tests for the averages obtained above.
    return compute_statistical_tests(overallVictories, overallScores, overallTimes, NUM_GAMES, dirs, alg_index)

def latex_table_results(all_victories_test, all_scores_test, all_times_test, ALG_NAMES, alg_index):

    # Build the results table
    prettyTable = build_results_table(all_victories_avg, all_victories_stErr, all_victories_test,
                     all_scores_avg, all_scores_stErr, all_scores_test,
                     all_timesteps_avg, all_timesteps_stErr, all_times_test,
                     SIGNIFICANCE_P_VALUE, games_repetitions, GAME_NAMES, NUM_GAMES, ALG_NAMES, ALG_NAMES_LETTER, alg_index)

    # Print the latex table.
    print_latex_table_results(prettyTable, NUM_GAMES, N_ALG, SIGNIFICANCE_P_VALUE)


def plots(dirs,txt):


    ALG_CONFIGS = ['Original']


    # drawPlot(overallVictories, dirs, 'Victories Percentage', 'Game',
    #                      'Victories', '13-16.pdf', ALG_NAMES, GAME_NAMES, False, True)

    #drawPlot(overallScores, dirs, 'Scores Averages - ' + txt, 'Algorithm',
    #                      'Scores', 'all_scores_all_'+txt+'.pdf', ALG_CONFIGS, ALG_NAMES, False, True)

def plotHeatMap(dirs, all_victories_test, all_scores_test, all_times_test):
    drawHeatMap(overallVictories, dirs, ALG_NAMES, ALG_NAMES_LETTER, GAME_NAMES, all_victories_avg, all_victories_stErr, all_victories_test,
                     all_scores_avg, all_scores_stErr, all_scores_test, SIGNIFICANCE_P_VALUE, CONFIG, HM_TITLE, HM_FILENAME, 'Blues')

def plotHeatMapGvA(dirs, all_victories_test, all_scores_test, all_times_test):
    drawHeatMapGvA(overallVictories, dirs, ALG_NAMES, ALG_NAMES_LETTER, GAME_NAMES, all_victories_avg, all_victories_stErr,
                all_victories_test,
                all_scores_avg, all_scores_stErr, all_scores_test, SIGNIFICANCE_P_VALUE, CONFIG, HM_TITLE, HM_FILENAME,
                'Blues')


def histograms(dirs,txt, alg_names):

    # drawHistogram(overallScores, dirs, 'Scores Averages', 'Algorithm',
    #                        'Scores', 'all_scores.pdf', GAME_NAMES, False, True)
    #
    drawHistogram(overallVictories, dirs, 'Victories Percentage (P = 1, L = 6)', 'Game',
                         'Win Rate', 'all_victories.pdf', GAME_NAMES, alg_names, False, True)


    # cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # game_names = [GAME_NAMES[i] for i in cols]
    #
    # newList = [[l[i] for i in cols] for l in overallVictories]
    # drawHistogram(newList, dirs, 'Victories Percentage', 'Algorithm',
    #                       'Victories', 'first.png', game_names, False, True)
    #
    # newList = [[l[i] for i in cols] for l in overallScores]
    # drawHistogram(newList, dirs, 'Scores Averages', 'Algorithm',
    #                      'Scores', 'second.png', game_names, False, True)
    #
    # cols = [3]
    # game_names = [GAME_NAMES[i] for i in cols]
    #
    # newList = [[l[i] for i in cols] for l in overallVictories]
    # drawHistogram(newList, dirs, 'Victories Percentage', 'Algorithm',
    #                       'Victories', 'thirds.pdf', game_names, False, True)
    #
    # newList = [[l[i] for i in cols] for l in overallScores]
    # drawHistogram(newList, dirs, 'Scores Averages', 'Algorithm',
    #                       'Scores', 'fourth', game_names, False, True)


def rankings(ALG_NAMES):

    games_gsu, all_points, points_per_game, tiebreaker, global_ranking, ranking_games, total_victories_avg\
        = compute_rankings(all_victories_avg, all_scores_avg, all_timesteps_avg, N_ALG, NUM_GAMES)
    prettySumTable, prettyGameTable = ranking_tables(global_ranking, ranking_games, games_gsu, total_victories_avg,
                                                     all_points, points_per_game, tiebreaker,
                                                     all_victories_stErr, all_scores_stErr, all_timesteps_stErr,
                                                     ALG_NAMES, GAME_NAMES)
    return prettySumTable, prettyGameTable


def comparison_table(points_table, ALG_NAMES):


    data_vict = (np.array(all_victories_avg)).mean(axis=0)
    data_vict_stdErr = (np.array(all_victories_stErr)).mean(axis=0)
    # dim1, dim2, table_avg, table_std = compute_table_victories(data_vict, data_vict_stdErr, ALG_NAMES)
    dim1, dim2, table_avg, table_std = compute_table_victories_budget(data_vict, data_vict_stdErr, ALG_NAMES)
    latex_table_array(dim1, dim2, table_avg, table_std, "Average Victories", True)

    points_dict = dict()
    sorted_points = []
    for row in points_table:
        points_dict[row[1]] = row[2]
    for alg in ALG_NAMES:
        sorted_points.append(points_dict[alg])

    # dim1, dim2, table, table_std = compute_table_victories(sorted_points, list(data_vict_stdErr), ALG_NAMES)
    dim1, dim2, table, table_std = compute_table_victories_budget(sorted_points, list(data_vict_stdErr), ALG_NAMES)
    latex_table_array(dim1, dim2, table, table_std, "Points", False)



def print_bench():

    results_list = ['results6']
    txt = "SampleMCTS"
    ALG_NAMES = ["SampleMCTS"]
    N_ALG = len(ALG_NAMES)

    results(results_list)
    for i in range(len(GAME_NAMES)):

        if len(all_victories_avg[i]) > 0:
            vict = (all_victories_avg[i][0]) * 100
            scor = (all_scores_avg[i][0])
        else:
            vict = "-"
            scor = "-"

        print GAME_NAMES[i], ",", vict, ",", scor



def printAll(results_list, txt, ALL_GAMES, ALG_NAMES):

    results(results_list)
    # all_victories_test, all_scores_test, all_times_test = stat_tests(results_list, alg_index)

    # histograms(results_list, txt, ALG_NAMES)
    # plots(results_list, txt)
    # plotHeatMap(results_list, all_victories_test, all_scores_test, all_times_test)
    # plotHeatMapGvA(results_list, all_victories_test, all_scores_test, all_times_test)

    # latex_table_results(all_victories_test, all_scores_test, all_times_test, ALG_NAMES, alg_index)
    prettySumTable, prettyGameTable = rankings(ALG_NAMES)
    latex_table_all_rankings(prettySumTable, NUM_GAMES)
    #latex_table_games_rankings(prettyGameTable, ALL_GAMES, GAME_NAMES)

    comparison_table(prettySumTable, ALG_NAMES)



if __name__ == "__main__":

    printAll(all_result_dirs, "Normal", GAME_NAMES, ALG_NAMES)



