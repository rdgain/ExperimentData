import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from src.config import *
import numpy as np


"""
Function to plot various visualizations for data. Saves all to files.
"""
def visualize(feature_file, feature_names, filename, sizex, sizey):
    # Read files into pandas data structure
    data = pandas.read_csv(feature_file, names=feature_names, usecols=range(len(feature_names)), delimiter=' ')

    # Scatter plot of the relationship between features
    axes = scatter_matrix(data)
    [s.xaxis.label.set_rotation(45) for s in axes.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in axes.reshape(-1)]
    [s.get_yaxis().set_label_coords(-2, 0.5) for s in axes.reshape(-1)]
    [s.set_xticks(()) for s in axes.reshape(-1)]
    [s.set_yticks(()) for s in axes.reshape(-1)]
    plt.gcf().subplots_adjust(left=0.15, bottom=0.2)
    plt.savefig("../plots/vis/team/" + filename + "_features_scatter.png")

    # Correlation between features heat map
    correlations = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(feature_names), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticklabels(feature_names)
    plt.tight_layout()
    plt.savefig("../plots/vis/team/" + filename + "_features_corr.png")

    # Box plot of each feature
    data.plot(kind='box', subplots=True, layout=(sizex, sizey), sharex=False, sharey=False)
    plt.tight_layout()
    plt.savefig("../plots/vis/team/" + filename + "_features_boxplot.png")

    # Density plot of each feature
    data.plot(kind='density', subplots=True, layout=(sizex, sizey), sharex=False, sharey=False, legend=False)
    plt.tight_layout()
    plt.savefig("../plots/vis/team/" + filename + "_features_density.png")

    # Histogram of each feature
    data.hist()
    plt.tight_layout()
    plt.savefig("../plots/vis/team/" + filename + "_features_hist.png")

    # Area plot of each feature
    data.plot(kind='area', subplots=True, layout=(sizex, sizey), sharex=False, sharey=False, legend=False)
    plt.tight_layout()
    plt.savefig("../plots/vis/team/" + filename + "_features_area.png")

    plt.show()


"""
Main function to access visualisation functions. Visualizes all data from CSV files.
"""
def visuals():
    names = ['CS', 'Gold', 'XP', 'Dmg']  # Set feature names
    names_all = []  # Collect team features

    # Plot data for each player
    for p in players:
        for n in names:
            names_all.append(p + " " + n)
        file = "../data/features_" + p + ".csv"
        visualize(file, names, p, 2, 2)

    # Plot team data
    file = "../data/features_team.csv"
    visualize(file, names_all, "team", 5, 4)

    # Plot all players together data
    file = "../data/features_all.csv"
    visualize(file, names, "all", 2, 2)
