__author__ = 'dperez'
import operator
from config import F1_RANKING_POINTS
import numpy as np



class GameScoreUnit:

    def __init__(self):
        pass

def gameScoreUnitComparator(gsu1, gsu2):

    if gsu1.win_avg > gsu2.win_avg:
        return -1

    if gsu1.win_avg < gsu2.win_avg:
        return 1

    if gsu1.score_avg > gsu2.score_avg:
        return -1

    if gsu1.score_avg < gsu2.score_avg:
        return 1

    if gsu1.timesteps_avg < gsu2.timesteps_avg:
        return -1

    if gsu1.timesteps_avg > gsu2.timesteps_avg:
        return 1

    return 0


def F1_helper(score):

    if score == 25: return pow(10, 10)
    if score == 18: return pow(10, 9)
    if score == 15: return pow(10, 8)
    if score == 12: return pow(10, 7)
    if score == 10: return pow(10, 6)
    if score == 8: return pow(10, 5)
    if score == 6: return pow(10, 4)
    if score == 4: return pow(10, 3)
    if score == 2: return pow(10, 2)
    if score == 1: return pow(10, 1)
    return 1


def compute_rankings(victories, scores, timesteps, num_alg, num_games):

    games_gsu = [[] for _ in range(num_games)]
    ranking_games = [[] for _ in range(num_games)]
    points_per_game = [[] for _ in range(num_alg)]
    all_victories_avg = [0 for _ in range(num_alg)]
    all_points = [0 for _ in range(num_alg)]
    tiebreaker_per_game = [[] for _ in range(num_alg)]
    tiebreaker = [0 for _ in range(num_alg)]

    global_rank = {}

    for game in range(num_games):

        #Take one game and compare all approaches.
        for approach in range(num_alg):

            gsu = GameScoreUnit()
            gsu.game_idx = game
            gsu.approach_idx = approach
            gsu.win_avg = victories[game][approach]
            gsu.score_avg = scores[game][approach]
            gsu.timesteps_avg = timesteps[game][approach]

            games_gsu[game].append(gsu)
            all_victories_avg[approach] += gsu.win_avg



        games_gsu[game].sort(gameScoreUnitComparator)

        ranking_game = {}
        for rank in range(num_alg):
            approach = games_gsu[game][rank].approach_idx
            points = F1_RANKING_POINTS[rank]
            points_per_game[approach].append(points)
            tiebreaker_per_game[approach].append(F1_helper(points))

            ranking_game[approach] = points

        sorted_x = sorted(ranking_game.items(), key=operator.itemgetter(1), reverse=True)
        ranking_games[game] = [int(i[0]) for i in sorted_x]

    for approach in range(num_alg):
        all_points[approach] = sum(points_per_game[approach])
        tiebreaker[approach] = sum(tiebreaker_per_game[approach])
        global_rank[approach] = all_points[approach]
        all_victories_avg[approach] /= num_games

    sorted_x = sorted(global_rank.items(), key=operator.itemgetter(1), reverse=True)
    global_ranking = [int(i[0]) for i in sorted_x]

    return games_gsu, all_points, points_per_game, tiebreaker, global_ranking, ranking_games, all_victories_avg


def ranking_tables(global_ranking, ranking_games, games_gsu, total_victories_avg, all_points, points_per_game,
                   tiebreaker, all_victories_stErr, all_scores_stErr, all_timesteps_stErr, alg_names, game_names):

    num_games = len(game_names)
    n_alg = len(alg_names)

    prettySumTable = [[] for _ in range(n_alg)]

    apprIdx = 0
    for approach in global_ranking:

        prettySumTable[apprIdx].append(apprIdx + 1)
        prettySumTable[apprIdx].append(alg_names[approach])
        prettySumTable[apprIdx].append(all_points[approach])
        stErr = np.average([[x[approach]] for x in all_victories_stErr]) # np.std(all_victories_stErr[:][approach]) / np.sqrt(len(all_victories_stErr[:][approach]))
        prettySumTable[apprIdx].append("%.2f (%.2f)" % (100.0*total_victories_avg[approach], 100.0*stErr))

        for game in range(num_games):
            prettySumTable[apprIdx].append(points_per_game[approach][game])

        apprIdx += 1


    prettyGameTable = []

    for game in range(num_games):

        prettyGameTableInd = [[] for i in range(n_alg)]

        apprIdx = 0
        for approach in ranking_games[game]:

            prettyGameTableInd[apprIdx].append(apprIdx + 1)
            prettyGameTableInd[apprIdx].append(alg_names[approach])
            prettyGameTableInd[apprIdx].append(points_per_game[approach][game])

            winPerc = "%.2f (%.2f)" % (100.0*games_gsu[game][apprIdx].win_avg, 100.0*all_victories_stErr[game][approach])
            prettyGameTableInd[apprIdx].append(winPerc)

            score = "%.2f (%.2f)" % (games_gsu[game][apprIdx].score_avg, all_scores_stErr[game][approach])
            prettyGameTableInd[apprIdx].append(score)

            timesteps = "%.2f (%.2f)" % (games_gsu[game][apprIdx].timesteps_avg, all_timesteps_stErr[game][approach])
            prettyGameTableInd[apprIdx].append(timesteps)

            apprIdx += 1

        prettyGameTable.append(prettyGameTableInd)


    return prettySumTable, prettyGameTable
