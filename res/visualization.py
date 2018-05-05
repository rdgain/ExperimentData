import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import setp
from config_classifier import *
import pylab


def feature_analysis():
    files = ["0-30", "30-70", "70-100"]

    global target_class_column, features

    for f in files:
        if f == "":  # full feature file, need to adjust features
            target_class_column = 20
            features = range(instance_column + 1, target_class_column)
            features.remove(9)
            features.remove(8)
            features.remove(7)
            features.remove(5)
        file = "data/train/resultdatalevrep" + f + ".txt"
        data = pandas.read_csv(file, names=feature_names, usecols=features, delimiter=' ')

        # axs = scatter_matrix(data)
        # [s.xaxis.label.set_rotation(45) for s in axs.reshape(-1)]
        # [s.yaxis.label.set_rotation(0) for s in axs.reshape(-1)]
        # [s.get_yaxis().set_label_coords(-1.3, 0.5) for s in axs.reshape(-1)]
        # [s.set_xticks(()) for s in axs.reshape(-1)]
        # [s.set_yticks(()) for s in axs.reshape(-1)]
        # plt.gcf().subplots_adjust(left=0.15, bottom=0.2)
        # plt.savefig("plots/features_" + f + "_scatter.png")

        # correlations = data.corr()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(correlations, vmin=-1, vmax=1)
        # fig.colorbar(cax)
        # ticks = np.arange(0, len(feature_names), 1)
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        # ax.set_xticklabels(feature_names)
        # ax.set_yticklabels(feature_names)
        # plt.tight_layout()
        # plt.savefig("plots/features_" + f + "_corr.png")

        # data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
        # plt.tight_layout()
        # plt.savefig("plots/features_" + f + "_boxplot.png")

        axs = data.plot(kind='density', subplots=True, layout=(4, 3), sharex=False, sharey=False, legend=True)
        [s.set_ylabel("") for s in axs.reshape(-1)]
        [s.legend(loc="upper right") for s in axs.reshape(-1)]
        plt.tight_layout()
        plt.savefig("plots/features_" + f + "_density.png")

        # data.hist()
        # plt.tight_layout()
        # plt.savefig("plots/features_" + f + "_hist.png")

        plt.show()


def plot_all_game_ticks():
    for g in test_games:
        if g != 44:
            continue
        # save to file
        print "Plotting game " + str(g)
        file1 = "data/predicted/mcts_" + str(g) + ".txt"
        file2 = "data/predicted/win_mcts_" + str(g) + ".txt"
        data = np.loadtxt(file1, comments='*', delimiter=' ')
        win = np.loadtxt(file2, comments='*', delimiter=' ')
        n_models = 4
        n_measures = 3  # f1, acc, err
        for m in range(n_models):
            model_f1 = data[m * n_measures]
            model_acc = data[m * n_measures + 1]
            model_error = data[m * n_measures + 2]
            x_axis = test_cases[0:len(model_f1)]
            plt.plot(x_axis, model_f1, label=model_names_p[m], color=model_colors[m], ls=model_style[m])
            # plt.fill_between(x_axis, [model_pred[i] - model_error[i] for i in range(len(model_pred))],
            #                  [model_pred[i] + model_error[i] for i in range(len(model_pred))],
            #                  alpha=0.2, antialiased=True)
        lgd = plt.legend(loc="upper right", fancybox=True, framealpha=0.7, fontsize=10, markerscale=0.2, ncol=1)
        plt.grid(True, linestyle='dotted')
        plt.ylabel("Model F1-score", fontsize=20)
        plt.xlabel("Game tick", fontsize=14)
        # plt.xticks(test_cases, [i/100 for i in test_cases])
        plt.title("Win average = " + ("%.2f" % win[0]) + " (+/- %.2f)" % win[1], fontsize=20)
        axes = plt.gca()
        axes.set_ylim([-0.1, 1.1])
        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axes.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        plt.tight_layout()
        plt.savefig("predictors/p/f1_mcts_" + str(g) + ".png")
        # plt.show()
        plt.close()


# main exec
global model_names, model_names_p, model_colors, models_test, model_style

if not TRAIN:  # only use some of the models
    model_names = [model_names[i] for i in range(len(model_names)) if i in models_test]
    last = model_names_p[-1]
    lastc = model_colors[-1]
    lasts = model_style[-1]
    model_names_p = [model_names_p[i] for i in range(len(model_names_p)) if i in models_test]
    model_colors = [model_colors[i] for i in range(len(model_colors)) if i in models_test]
    model_style = [model_style[i] for i in range(len(model_style)) if i in models_test]
    if INCLUDE_RULE_MODEL:
        model_names_p.append(last)
        model_colors.append(lastc)
        model_style.append(lasts)

plot_all_game_ticks()
