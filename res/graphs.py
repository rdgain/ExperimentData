__author__ = 'dperez'

import matplotlib.pyplot as plt
import pylab
import numpy as np
import seaborn as sns
from config import *


def drawHistogram(all_data, result_dirs, title, xlab, ylab, outputFile, game_names, alg_names, showPlot=True, saveToFile=True):

    # print "Writing to ", outputFile

    #Create a figure
    fig = pylab.figure()
    fig.set_canvas(plt.gcf().canvas)

    #Add a subplot (Grid of plots 1x1, adding plot 1)
    ax = fig.add_subplot(111)

    width = 0.15
    colors = ['yellow','green','red','black','w','black']
    hatches = ['//',':','//','.','//',':']
    n_alg = len(result_dirs)
    ind = np.arange(len(all_data[0]))
    n_games = len(all_data[0])

    avg_plot = []
    err_plot = []
    StartingGamePlot = 0
    n_algorithms = len(result_dirs)

    for i in range(n_algorithms):
        avg_plot.append([0 for x in range(n_games)])
        err_plot.append([0 for x in range(n_games)])
        for j in range(n_games):
            #j = j + n_games/2
                avg_plot[i][j] = np.average(all_data[i][j])
                err_plot[i][j] = np.std(all_data[i][j]) / np.sqrt(len(all_data[i][j]))
            # if j < 17 :
            #     avg_plot[i][j - n_games/2] = np.average(all_data[i][j])
            #     err_plot[i][j - n_games/2] = np.std(all_data[i][j]) / np.sqrt(len(all_data[i][j]))
            # if j > 17 :
            #     avg_plot[i][j - n_games/2 -1] = np.average(all_data[i][j])
            #     err_plot[i][j - n_games/2 -1] = np.std(all_data[i][j]) / np.sqrt(len(all_data[i][j]))
                

    rects = [0 for x in range(n_alg)]
    for i in range(n_alg):
        rects[i] = ax.bar(ind+width*i , avg_plot[i], width,
                          color=colors[i], yerr=err_plot[i],
                          edgecolor='black', hatch=hatches[i], ecolor='black', label=alg_names[i])


    ax.set_xticks(ind+width*2)


    ax.yaxis.grid(True)
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

    maps_txt = []
    for j in range(n_games):
        #ss = 'Map ' + str(StartingMapPlot+j+1)
        #j = j + n_games
        #if j != 17 :
            ss = game_names[StartingGamePlot+j]
            maps_txt.append(ss)
            

    ax.set_xticklabels(maps_txt)

    #Titles and labels
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    #plt.xlim([8,22])
    #plt.ylim([0,200]) #175

    plt.legend(alg_names)
    plt.legend(bbox_to_anchor=(0.05, 0.8, 0.9, .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)

    # plt.xlim([8,22])
    plt.ylim([0, 1.4])  # 175

    fig.set_size_inches(15,5)

    if saveToFile:
        fig.savefig(outputFile)

    # And show it:
    if showPlot:
        plt.show()


def drawPlot(all_data, result_dirs, title, xlab, ylab, outputFile, alg_configs, alg_names, showPlot=True, saveToFile=True):

    #Create a figure
    fig = pylab.figure()

    #Add a subplot (Grid of plots 1x1, adding plot 1)
    ax = fig.add_subplot(111)

    num_configs = len(all_data) #algorithms
    num_algs = len(all_data[0]) #games

##    alg_index = [0, 1, 2] #1-6
##    alg_index = [3, 4, 5] #2-8
##    alg_index = [6, 7, 8] #5-10
##    alg_index = [9, 10, 11] #7-12
##    alg_index = [12, 13, 14] #10-14
    alg_index = [0, 1, 2] #13-16
##    alg_index = [0, 3, 6, 9, 12, 15] #vanilla
##    alg_index = [1, 4, 7, 10, 13, 16] #OneStep
##    alg_index = [2, 5, 8, 11, 14, 17] #MCTS
##    alg_index = alg_configs #all algorithms

    # width = 0.15
    # colors = ['w','grey','w','black','w','black']
    # hatches = ['//',':',':','.','//',':']
    # n_alg = len(result_dirs)
    # ind = np.arange(len(all_data[0]))
    # n_games = len(all_data[0])
    #
    avg_plot = []
    err_plot = []
    line_styles=['-','--','-.',':','dotted','dashdot']

    # StartingGamePlot = 0
    # n_algorithms = len(result_dirs)


    for i in range(len(alg_index)):
         avg_plot.append([0 for x in range(num_algs)])
         err_plot.append([0 for x in range(num_algs)])
         for j in range(num_algs):
             avg_plot[i][j] = np.average(all_data[alg_index[i]][j])
             err_plot[i][j] = np.std(all_data[alg_index[i]][j]) / np.sqrt(len(all_data[alg_index[i]][j]))

##    for i in range(num_algs):
##        avg_plot.append([0 for x in range(num_configs)])
##        err_plot.append([0 for x in range(num_configs)])
##        for j in range(num_configs):
##            avg_plot[i][j] = np.average(all_data[j][i])
##            err_plot[i][j] = np.std(all_data[j][i]) / np.sqrt(len(all_data[j][i]))



    rects = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    for i in range(len(alg_index)):        
        ax.errorbar(rects, avg_plot[i], yerr=err_plot[i], label=alg_configs[alg_index[i]], linewidth=2)
                          # , color=colors[i], linestyle=line_styles[i],
                          # edgecolor='black', hatch=hatches[i], ecolor='black')


    # ax.set_xticks(ind+width*2)
    # ax.yaxis.grid(True)
    # ax.set_xlabel('xlabel')
    # ax.set_ylabel('ylabel')
    #
    # maps_txt = []
    # for j in range(n_games):
    #     #ss = 'Map ' + str(StartingMapPlot+j+1)
    #     ss = game_names[StartingGamePlot+j]
    #     maps_txt.append(ss)
    #
    plt.xticks(rects)
    ax.set_xticklabels(alg_names)

    #Titles and labels
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.legend(alg_configs)
    plt.legend(bbox_to_anchor=(0.05, 0.8, 0.9, .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

    #plt.xlim([8,22])
    plt.ylim([0,1.4]) #175

    # fig.set_size_inches(15,5)

    if saveToFile:
        fig.savefig(outputFile)

    # And show it:
    if showPlot:
        plt.show()


##
# Plots N algorithms in NxN heatmap
# (values represent in how many games row was significantly better than column)
# saves heatmap to file
##
def drawHeatMap(all_data, result_dirs, alg_names, alg_letters, game_names, all_victories_avg, all_victories_stErr, all_victories_test,
                     all_scores_avg, all_scores_stErr, all_scores_test, SIGNIFICANCE_P_VALUE, CONFIG, title, txt, colour):

    num_games = len(all_data[0]) #games
    num_algs = len(all_data) #algorithms

    stat_data_vic = [[0 for i in range(num_algs)] for j in range(num_algs)]
    stat_data_sc = [[0 for i in range(num_algs)] for j in range(num_algs)]
    mask = np.zeros_like(stat_data_vic)
    annotations = [['' for i in range(num_algs)] for j in range(num_algs)]
    annotations = np.array(annotations)

    max_vic = 0
    max_sc = 0

    for game in range(len(all_victories_test)) :
        for i in range(num_algs) :
            stat_data_vic[i][i] = -1
            stat_data_sc[i][i] = -1
            annotations[i][i] = 'x'
            for j in range(i,num_algs) :
                mask[i][j] = True
                mask[j][i] = True

                #victories
                if all_victories_test[game][i][j] != 0 :
                    if all_victories_test[game][i][j] < SIGNIFICANCE_P_VALUE :
                        if all_victories_avg[game][i] > all_victories_avg[game][j]:
                            stat_data_vic[i][j] += 1
                        else:
                            stat_data_vic[j][i] += 1

                if (stat_data_vic[i][j] > max_vic):
                    max_vic = stat_data_vic[i][j]
                if (stat_data_vic[j][i] > max_vic):
                    max_vic = stat_data_vic[j][i]

                #scores
                if all_scores_test[game][i][j] != 0 :
                    if all_scores_test[game][i][j] < SIGNIFICANCE_P_VALUE :
                        if all_scores_avg[game][i] > all_scores_avg[game][j]:
                            stat_data_sc[i][j] += 1
                        else:
                            stat_data_sc[j][i] += 1

                if (stat_data_sc[i][j] > max_sc):
                    max_sc = stat_data_sc[i][j]
                if (stat_data_sc[j][i] > max_sc):
                    max_sc = stat_data_sc[j][i]


            mask[i][i] = False


    # vics = False
    vics = True
    scores = not vics

    #set label and title sizes for figure
    labelsize = 20
    titlesize = 30

    #Possible heatmap color values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r,
    # CMRmap, CMRmap_r,
    # Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn,
    # PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r,
    # PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,
    # RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r,
    # Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r,
    # autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r,
    # copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r,
    # gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r,
    # gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r,
    # inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink,
    # pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r,
    # spring, spring_r, summer, summer_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r
    # Check here: http://matplotlib.org/users/colormaps.html


    if vics :

        #custom heatmap with discrete values (remove max_vic to turn continuous)
        cmap = plt.cm.get_cmap(colour, max_vic)

        #plot heatmap
        ax1 = sns.heatmap(stat_data_vic, cmap=cmap, linewidths=.1, vmin=0) #,cbar_kws={'label': 'Number of games'}

        #grey out diagonal
        ax1 = sns.heatmap(stat_data_vic, mask=mask, cmap=plt.cm.Greys, linewidths=.1, cbar=False)

        #print alg_names so you know what's in there
        print alg_names

        #use letters for labels
        ax1.set_xticklabels(alg_letters)
        ax1.set_yticklabels(list(reversed(alg_letters)))

        #format tick labels
        plt.xticks(rotation=0, fontsize = labelsize, fontweight='bold')
        plt.yticks(rotation=0, fontsize = labelsize, fontweight='bold')

        #tighten resulting image around plot
        fig1 = ax1.get_figure()
        plt.tight_layout(rect=[0, 0, 1.1, 0.95])

        #custom colorbar ticks and labels
        cbar = ax1.collections[0].colorbar
        ticks = []
        for i in range(max_vic) :
            ticks.append(i + 0.5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(range(max_vic))
        cbar.ax.tick_params(labelsize=labelsize)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")

        #set figure title
        ax1.set_title(title, fontweight='bold', fontsize=titlesize, y=1.01)  # increase or decrease y as needed

        #save figure
        fig1.savefig("victories-" + CONFIG + txt + ".pdf")
        # fig1.savefig("victories.pdf")
        # plt.show()

    #########################

    if scores :

        #custom heatmap with discrete values (remove max_sc to turn continuous)
        cmap = plt.cm.get_cmap(colour, max_sc)

        #plot heatmap
        ax1 = sns.heatmap(stat_data_sc, cmap=cmap, linewidths=.1, vmin=0) #,cbar_kws={'label': 'Number of games'}

        #grey out diagonal
        ax1 = sns.heatmap(stat_data_sc, mask=mask, cmap=plt.cm.Greys, linewidths=.1, cbar=False)

        #print alg_names so you know what's in there
        print alg_names

        #use letters for labels
        ax1.set_xticklabels(alg_letters)
        ax1.set_yticklabels(list(reversed(alg_letters)))

        #format tick labels
        plt.xticks(rotation=0, fontsize = labelsize, fontweight='bold')
        plt.yticks(rotation=0, fontsize = labelsize, fontweight='bold')

        #tighten resulting image around plot
        fig1 = ax1.get_figure()
        plt.tight_layout(rect=[0, 0, 1.1, 0.95])

        #custom colorbar ticks and labels
        cbar = ax1.collections[0].colorbar
        ticks = []
        for i in range(max_sc) :
            ticks.append(i + 0.5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(range(max_sc))
        cbar.ax.tick_params(labelsize=labelsize)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")

        #set figure title
        ax1.set_title(title, fontweight='bold', fontsize=titlesize, y=1.01)  # increase or decrease y as needed

        #save figure
        fig1.savefig("scores-" + CONFIG + txt + ".pdf")
        # fig1.savefig("scores.pdf")
        # plt.show()


##
# Plots N algorithms in NxN heatmap
# (values represent in how many games row was significantly better than column)
# saves heatmap to file
##
def drawHeatMapGvA(all_data, result_dirs, alg_names, alg_letters, game_names, all_victories_avg, all_victories_stErr, all_victories_test,
                     all_scores_avg, all_scores_stErr, all_scores_test, SIGNIFICANCE_P_VALUE, CONFIG, title, txt, colour):

    vics = False
    scores = not vics

    #set label and title sizes for figure
    labelsize = 5
    # plt.figsize = (10,1)

    if vics :

        # custom heatmap with discrete values (remove max_vic to turn continuous)
        cmap = plt.cm.get_cmap(colour)  # , max_vic)

        # plot heatmap
        cm = sns.clustermap(all_expG_avg, cmap=cmap, linewidths=.1, vmin=0, yticklabels=game_names,
                            xticklabels=alg_names) #,cbar_kws={'label': 'Number of games'}
        hm = cm.ax_heatmap.get_position()
        plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=labelsize)
        plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), fontsize=labelsize)
        cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*0.25, hm.height])
        col = cm.ax_col_dendrogram.get_position()
        cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.25, col.height*0.5])

        #save figure
        plt.savefig("clustermap-expgame.pdf")
        # plt.show()

    #########################

    if scores :

        #custom heatmap with discrete values (remove max_sc to turn continuous)
        cmap = plt.cm.get_cmap(colour)

        #plot heatmap
        cm = sns.clustermap(all_scores_avg, cmap=cmap, linewidths=.1, vmin=0, yticklabels=game_names,
                            xticklabels=alg_names) #,cbar_kws={'label': 'Number of games'}
        hm = cm.ax_heatmap.get_position()
        plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=labelsize)
        plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), fontsize=labelsize)
        cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*0.25, hm.height])
        col = cm.ax_col_dendrogram.get_position()
        cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.25, col.height*0.5])

        #save figure
        plt.savefig("clustermap-scores1.pdf")


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        # print(v)
        plotMat.append(v)

    # print('plotMat: {0}'.format(plotMat))
    # print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

