import matplotlib.pyplot as plt
from src.config import *
import numpy as np
from numpy.linalg import norm
from scipy import stats


"""
Function to plot a matrix, where each row is a different player and each column is the corresponding value in 1 game.
Saves the plots to .png files in the ``plots'' directory.
"""
def plotPlayerData(dataArray, ylabeltxt, figtxt, color):
    for p in range(no_players):
        plt.plot(dataArray[p], color)
        plt.ylabel(ylabeltxt)
        plt.title(players[p])
        plt.savefig("../plots/" + players[p] + "_" + figtxt + ".png")
        plt.show()


"""
Function to plot a matrix, where each row is a different player and each column is the corresponding value in 1 game,
against an array where each value corresponds to the team's total of the same feature.
Saves the plots to .png files in the ``plots'' directory.
"""
def plotPlayerTeamData(playerArray, teamArray, labeltxt, figtxt, color, winArray):
    colors = [color] * len(teamArray)  # set color array to color passed

    if color is None and winArray is not None:  # if win array is passed, use this to determine color instead
        colors = []
        for i in range(len(win)):
            if win[i] == 'Win':
                colors.append('#DC143C')
            else:
                colors.append('#3CB371')

    for p in range(no_players):  # 1 plot for each player role
        if color is None:
            for i in range(len(teamArray)):  # plot each point separately to give correct colors
                plt.plot(playerArray[p][i], teamArray[i], 'x', color=colors[i])
        else:  # plot all points together for speed of execution
            plt.plot(playerArray[p], teamArray, 'x', color=color)

        # add a trend line
        z = np.polyfit(playerArray[p], teamArray, 1)
        k = np.poly1d(z)
        plt.plot(playerArray[p], k(playerArray[p]))

        # calculate distribution statistics projected onto the trend line
        # p1 = np.array([z[0], k(playerArray[p])[0]])
        # p2 = np.array([z[-1], k(playerArray[p])[-1]])
        # distArray = []
        # for point in range(len(playerArray[p])):
        #     p3 = np.array([playerArray[p][point], teamArray[point]])
        #     d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
        #     distArray.append(d)
        # print(stats.describe(distArray))

        plt.ylabel('Team End Game ' + labeltxt)
        plt.xlabel(players[p] + ' ' + labeltxt)
        plt.title(players[p])
        plt.grid(ls='--', color='lightgray')
        plt.savefig("../plots/" + players[p] + "_" + figtxt + "_" + labeltxt + ".pdf")
        plt.show()
