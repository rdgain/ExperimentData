from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import random
import os
from os import listdir
from os.path import isfile, join
from scipy import stats
from scipy import optimize
import numpy.ma as ma
import scipy
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import pylab
from collections import Counter
from ntbea import *
from math import sqrt

file_path = "exp6"  # Seeded with sota, 1500 iterations
# file_path = "test"
n_games = 20
parameters = ["crossover_type",
              "population_size",
              "frame_skip_type",
              "mc_rollouts_repeat",
              "no_elites",
              "genetic_operator",
              "fitness_assignment",
              "init_type",
              "frame_skip",
              "selection_type",
              "shift_buffer",
              "dynamic_depth",
              "diversity_weight",
              "offspring_count",
              "shift_discount",
              "mutation_type",
              "individual_length",
              "mc_rollouts_length_perc"]
param_space = {"crossover_type": ["UNIFORM", "1_POINT", "2_POINT"],
               "population_size": [1, 10, 15, 20],
               "frame_skip_type": ["REPEAT", "NULL", "RANDOM", "SEQUENCE"],
               "mc_rollouts_repeat": [1, 5, 10],
               "no_elites": [0, 1],
               "genetic_operator": ["MUTATION_AND_CROSSOVER", "MUTATION_ONLY", "CROSSOVER_ONLY"],
               "fitness_assignment": ["LAST", "DELTA", "AVG", "MIN", "MAX", "DISCOUNT"],
               "init_type": ["RANDOM", "1SLA", "MCTS"],
               "frame_skip": [0, 5, 10],
               "selection_type": ["RANK", "TOURNAMENT", "ROULETTE"],
               "shift_buffer": ["false", "true"],
               "dynamic_depth": ["false", "true"],
               "diversity_weight": [0.0, 0.5, 1.0],
               "offspring_count": [5, 10, 15, 20],
               "shift_discount": [0.9, 0.99, 1.0],
               "mutation_type": ["UNIFORM", "1-BIT", "3-BITS", "SOFTMAX", "DIVERSITY"],
               "individual_length": [5, 10, 15, 20],
               "mc_rollouts_length_perc": [0.0, 0.5, 1.0, 2.0]}
dependencies = {"mc_rollouts_repeat": ["mc_rollouts_length_perc", 0.5, 1.0, 2.0],
                "mutation_type": ["genetic_operator", "MUTATION_AND_CROSSOVER", "MUTATION_ONLY"],
                "shift_discount": ["shift_buffer", "true"],
                "diversity_weight": ["mutation_type", "DIVERSITY"],
                "frame_skip_type": ["frame_skip", 5, 10],
                "selection_type": ["genetic_operator", "MUTATION_AND_CROSSOVER", "CROSSOVER_ONLY"],
                "crossover_type": ["genetic_operator", "MUTATION_AND_CROSSOVER", "CROSSOVER_ONLY"]}
old_win_rate = [0, 0, 0, 100, 4, 9, 5, 36, 3, 19, 65, 40, 30, 63, 98, 58, 98, 100, 95, 100]
game_mapping = ["Dig Dug",
                "Lemmings",
                "Roguelike",
                "Chopper",
                "Crossfire",
                "Chase",
                "Camel Race",
                "Escape",
                "Hungry Birds",
                "Bait",
                "Wait for Breakfast",
                "Survive Zombies",
                "Modality",
                "Missile Command",
                "Plaque Attack",
                "Sea Quest",
                "Infection",
                "Aliens",
                "Butterflies",
                "Intersection"
                ]


def process_file(f_name):
    tuples1_values = [[] for _ in range(len(parameters))]
    tuples2_values = {}
    with open(file_path + "/" + f_name) as f:
        lines = f.readlines()
        for line in lines:
            if "Solution evaluated" in line:
                split_line = line.split(':')
                split_line2 = split_line[1].split(']')
                solution_str = split_line2[0].strip()[1:].replace(',', '').split()
                q = float(split_line2[1].strip())
                solution = []
                if q > -1:
                    for s in range(len(solution_str)):
                        param = int(solution_str[s])
                        solution.append(param)

                        if check_param_dependency(s, solution_str):
                            tuples1_values[s].append((param, q))

                        for p in range(len(solution_str)):
                            if s != p:
                                param2 = int(solution_str[p])
                                if check_param_dependency(s, solution_str) and check_param_dependency(p, solution_str):
                                    if (s, p) not in tuples2_values:
                                        tuples2_values[(s, p)] = []
                                    tuples2_values[(s, p)].append((param, param2, q))

    return tuples1_values, tuples2_values


def check_param_dependency(param_idx, solution_str):
    """
    Checks if the given parameter in the solution affects phenotype, returns true if so, false otherwise, depending
    on parameter dependencies
    """
    if parameters[param_idx] in dependencies:
        depend = dependencies[parameters[param_idx]]
        depend_idx = parameters.index(depend[0])
        depend_value = param_space[depend[0]][int(solution_str[depend_idx])]
        if depend_value in depend[1:]:
            return True
        else:
            return False
    else:
        return True


def process_file_fitness(f_name):
    fitness = []
    fit_std = []

    best = []
    all_s = []

    sota = []
    sota_std = []

    ss = SearchSpace("space", len(param_space))
    n_tuple = NTupleLandscape(ss, tuple_config=[1, 2, ss.get_num_dims()], ucb_epsilon=sqrt(2))
    n_tuple.init()

    with open(file_path + "/" + f_name) as f:
        lines = f.readlines()
        s_sota = np.array([])
        for i in range(len(lines)):
            line = lines[i]
            if "Solution evaluated" in line:
                split_line = line.split(':')
                split_line2 = split_line[1].split(']')
                solution_str = split_line2[0].strip()[1:].replace(',', '').split()
                all_s.append(', '.join(solution_str))

                point = np.array([int(x) for x in solution_str])
                q = float(split_line2[1].strip())
                n_tuple.add_evaluated_point(point, q)

                if s_sota.size == 0:
                    s_sota = point
                mean, std = n_tuple.get_mean_estimtate(s_sota)
                sota.append(mean)
                sota_std.append(std)

                best_solution = lines[i + 1].split(':')[1].split(']')[0].strip()[1:]

                if q > -1:
                    mean, std = n_tuple.get_mean_estimtate(point)
                    fitness.append(mean)
                    fit_std.append(std)
                    best.append(0 if ', '.join(solution_str) != best_solution else 1)
                else:
                    print(split_line2[0].strip() + "]")
            # if "Solution returned" in line:
            #     best_solution = line.split(':')[1].split(']')[0].strip()[1:]
            #     eval = 0
            #     for s in all_s:
            #         if s == best_solution:
            #             eval += 1
            #     print(f_name, eval)

    return fitness, fit_std, best, sota, sota_std


def plot(values):
    plt.figure()
    max_param = 0
    for par in values:
        if par[0] > max_param:
            max_param = par[0]
    proc = [[] for _ in range(max_param + 1)]
    for par in values:
        proc[par[0]].append(par[1])
    # print(proc)

    for idx in range(len(proc)):
        x = [idx + random.uniform(-0.2, 0.2) for _ in range(len(proc[idx]))]
        plt.scatter(x, proc[idx])
    plt.show(block=False)
    # print(values)


def plot_1tuples_per_game(only_files):
    print("Plotting 1-tuples per game ...")
    for f in only_files:
        param_values, tuples2 = process_file(f)
        # for params in param_values:
        #     plot(params)
        name = f.split(".")[0]

        plot_path = "1tuples-plots-" + file_path
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        fig = plt.figure(figsize=(15, 15))
        x = []
        y = []
        z = []
        for i in range(len(param_values)):
            ax = fig.add_subplot(5, 5, i + 1)
            for value, fitness in param_values[i]:
                x.append(i)
                y.append(value)
                z.append(fitness)
            min_fitness = min(z) + 0.1
            max_fitness = max(z) + 0.1
            for j in range(len(y)):
                fit = z[j] + 0.1
                # c = (abs((fit - max_fitness) / max(fit, max_fitness)), abs((fit - min_fitness) / max(fit, min_fitness)), 0)
                # ax.scatter(y[j], z[j], alpha=0.1, color=c)
                ax.scatter(y[j], z[j], alpha=0.1, color='b')
            ax.set_xticks(list(set(y)))
            ax.set_xticklabels(param_space[parameters[i]])
            ax.set_yticks(list(set(z)))
            ax.set_title(parameters[i])
            if any(isinstance(x, str) for x in param_space[parameters[i]]):  # rotate strings so labels show nicer
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
            x = []
            y = []
            z = []
        plt.tight_layout()
        plt.savefig(plot_path + "/" + name + ".png")
        plt.close(fig)
        # plt.show()


def plot_1tuples_all_games(only_files):
    print("Plotting 1-tuples all games ...")

    plot_path = "1tuples-" + file_path
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    tuples1 = {}
    for f in only_files:
        game = int(f.split('_')[2])
        tuples1[game], _ = process_file(f)

    games = range(n_games)
    # games = [3, 13, 15, 19]  # mc rollouts perc
    # games = [3, 13, 14]  # genetic operator
    # games = [3, 11, 13, 14, 15, 18]  # offspring count
    # games = [6, 10, 14, 19]  # frame skip type

    params = range(len(parameters))
    # params = [17]  # mc rollouts perc
    # params = [5]  # genetic operator
    # params = [13]  # offspring count
    # params = [2]   # frameskip type

    for param in params:
        # new figure for each param showing its plot in each of the 20 games
        fig, axes = plt.subplots(5, 4, figsize=(15, 15))
        axes = axes.flatten()
        # fig = plt.figure(figsize=(15, 15))
        for game in games:
            if game > -1:
                all_values = tuples1[game]
                ax = axes[games.index(game)]  # fig.add_subplot(4, 5, game+1, sharex=True)
                ax.grid(ls='--', color='lightgray')
                x = []
                y = []
                for value, fitness in all_values[param]:
                    x.append(value)
                    y.append(fitness)

                n = len(param_space[parameters[param]])
                # data = [[0 for _ in range(n+2)] for _ in range(6)]
                # maxx = 100
                # for j in range(len(x)):
                #     if data[int(y[j]/0.2)][x[j]+1] < maxx:
                #         data[int(y[j]/0.2)][x[j]+1] += 1
                #
                # # Normalize data?
                # minv = data[0][0]
                # maxv = data[0][0]
                # for i in range(len(data)):
                #     for j in range(len(data[i])):
                #         if data[i][j] < minv:
                #             minv = data[i][j]
                #         if data[i][j] > maxv:
                #             maxv = data[i][j]
                # for i in range(len(data)):
                #     for j in range(len(data[i])):
                #         data[i][j] = (data[i][j] - minv) * 1.0 / (maxv - minv)
                #
                # im = ax.imshow(data, interpolation='spline16', extent=[-1.5, n+0.5, -0.1, 1.1], aspect=n,
                #                origin='lower', cmap='copper', vmin=0.0, vmax=1.0)

                # for i in range(len(x)):
                #     ax.scatter(x[i], y[i], alpha=0.01, color='b')

                data = [[[0.0 for _ in range(4)]] for _ in range(n)]

                max_fitness = 1.01
                for i in range(n):
                    fit = np.average([y[j] for j in range(len(y)) if x[j] == i])
                    v = abs((fit - max_fitness) / max(fit, max_fitness))
                    data[i][0][0] = (1.0 if fit == 0 else v)
                    data[i][0][1] = 1.0  # (1.0 if fit == 0 else v)  # abs((fit - min_fitness) / max(fit, min_fitness))
                    data[i][0][2] = 1.0  # (1.0 if fit == 0 else v)
                    data[i][0][3] = (1.0 if fit == 0 else 0.5)

                im = ax.imshow(data, interpolation='spline16', cmap='RdYlGn', vmin=0.0, vmax=1.0, origin='lower')

                weight_counter = Counter(x)
                s_weights = [weight_counter[x[i]]*0.7 for i, _ in enumerate(x)]
                ax.scatter([0]*len(x), x, color='black', s=s_weights)

                ax.set_yticks([x for x in range(n)])
                ax.set_yticklabels([x for x in range(n)])
                # ax.set_yticklabels(param_space[parameters[param]])
                ax.set_ylim(-0.5, n-0.5)

                # ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xticks([])
                ax.set_title(str(game) + " " + game_mapping[game], size=18)
                # ax.label_outer()

        for ax in axes:
            if any(isinstance(x, str) for x in param_space[parameters[param]]):  # rotate strings so labels show nicer
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=15)
            plt.setp(ax.yaxis.get_majorticklabels(), fontsize=15)

        # cbar = fig.colorbar(im, extend="both", ax=axes)
        # cbar.set_label(label="normalized number of occurrences", size=18)

        # fig.text(0.5, 0.37, "Parmeter Values", ha='center', va='center', size=20)
        # fig.text(0.06, 0.7, "fitness", ha='center', va='center', rotation='vertical', size=20)

        plt.tight_layout(rect=[0.06, 0.3, 1, 1])
        plt.savefig(plot_path + "/" + parameters[param] + ".png")
        # plt.show()
        plt.close(fig)


def plot_2tuples_all_games(only_files):
    print("Plotting 2-tuples all games ...")

    plot_path = "2tuples-" + file_path
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    tuples2 = {}
    for f in only_files:
        game = int(f.split('_')[2])
        _, tuples2[game] = process_file(f)

    pars1 = parameters
    pars2 = parameters
    # pars2 = ["mutation_type"]
    # pars1 = ["crossover_type"]
    # pars1 = ["population_size"]
    # pars2 = ["individual_length"]

    for p1 in pars1:
        for p2 in pars2:
            if p1 != p2:
                # One figure per pair
                fig, axs = plt.subplots(4, 5, figsize=(15, 15))
                axes = axs.flatten()
                for game in range(n_games):
                    # One plot per game
                    if game > -1:
                        all_values = tuples2[game]
                        ax = axes[game]  # fig.add_subplot(4, 5, game+1, sharex=True)
                        ax.set_adjustable('box')
                        ax.grid(ls='--', color='lightgray')
                        ax.set_facecolor('black')

                        x = []
                        z = []

                        x_ = []
                        y_ = {}  # indexed by (x_, z_)
                        z_ = []

                        for value1, value2, fitness in all_values[(parameters.index(p1), parameters.index(p2))]:
                            x.append(value1)
                            z.append(value2)

                            if (value1, value2) not in y_:
                                x_.append(value1)
                                z_.append(value2)
                                y_[(value1, value2)] = []
                            y_[(value1, value2)].append(fitness)

                        data = [[[0.0 for _ in range(4)] for _ in range(len(param_space[p1]))] for _ in range(len(param_space[p2]))]

                        min_fitness = 0.01
                        max_fitness = 1.01
                        for j in range(len(x_)):
                            fit = np.average(y_[(x_[j], z_[j])])
                            v = abs((fit - max_fitness) / max(fit, max_fitness))
                            data[z_[j]][x_[j]][0] = (1.0 if fit == 0 else v)
                            data[z_[j]][x_[j]][1] = 1.0  # (1.0 if fit == 0 else v)  # abs((fit - min_fitness) / max(fit, min_fitness))
                            data[z_[j]][x_[j]][2] = 1.0  # (1.0 if fit == 0 else v)
                            data[z_[j]][x_[j]][3] = 1.0

                        im = ax.imshow(data, interpolation='spline16', cmap='RdYlGn', vmin=0.0, vmax=1.0, origin='lower')

                        # counts = {}
                        # for i in range(len(x)):
                        #     for j in range(len(z)):
                        #         if (i, j) not in counts:
                        #             counts[(i, j)] = 0
                        #         counts[(i, j)] += 1
                        # for (i, j) in counts:
                        #     v = counts[(i, j)]
                        #     ax.scatter(i, j, color='black', alpha=0.005*v, s=0.5*v)
                        # ax.scatter(x, z, alpha=0.005, color='black', s=100)
                        combos = list(zip(x, z))
                        weight_counter = Counter(combos)
                        s_weights = [weight_counter[(x[i], z[i])] for i, _ in enumerate(x)]
                        a_weights = [weight_counter[(x[i], z[i])]*0.005 for i, _ in enumerate(x)]
                        ax.scatter(x, z, color='black', s=s_weights)

                        ax.set_ylim(-0.5, len(param_space[p2])-0.5)

                        xticks = list(range(len(param_space[p1])))
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(param_space[p1])

                        yticks = list(range(len(param_space[p2])))
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(param_space[p2])

                        ax.set_title(str(game) + " " + game_mapping[game], size=20)
                        ax.label_outer()

                for ax in axes:
                    if any(isinstance(x, str) for x in param_space[p1]):  # rotate strings so labels show nicer
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
                    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=16)
                    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=16)

                # cbar = fig.colorbar(im, extend="both", ax=axes)
                # cbar.set_label(label="fitness", size=18)

                # outer axes labels
                p1_label = p1.replace("_", " ").capitalize()
                p2_label = p2.replace("_", " ").capitalize()  # p2.split("_")[0].capitalize() + " " + p2.split("_")[1].capitalize()
                fig.text(0.5, 0.04, p1_label, ha='center', va='center', size=28)
                fig.text(0.03, 0.5, p2_label, ha='center', va='center', rotation='vertical', size=28)

                plt.tight_layout(rect=[0.06, 0.045, 1, 1])
                plt.savefig(plot_path + "/" + p1 + "-" + p2 + ".png")
                plt.close(fig)


def win_rate_analysis(only_files):
    print("game\t\t\t\tnew %\told %\t\ttotal %\t\tsolution")
    for f in only_files:
        game = int(f.split('_')[2])
        with open(file_path + "/" + f) as fl:
            lines = fl.readlines()
            win_rate_g = -1
            win_rate_g_sterr = -1
            win_rate_total = -1
            win_rate_total_sterr = -1
            solution = ""
            win_rates_game = []
            win_rates_total = []
            counting_total = False
            counting_game = False
            for line in lines:
                if "Solution returned" in line:
                    counting_game = True
                    solution = line.split(":")[-1].strip()
                    continue
                if "Solution fitness" in line:
                    # win_rate_g = round(np.average(win_rates_game)*100, 2)
                    win_rate_g = np.average(win_rates_game) * 100
                    win_rate_g_sterr = round(np.std(win_rates_game) * 100 / np.sqrt(len(win_rates_game)), 3)
                    counting_game = False
                    counting_total = True
                    continue
                if "All games" in line:
                    win_rate_total = round(np.average(win_rates_total) * 100, 2)
                    win_rate_total_sterr = round(np.std(win_rates_total) * 100 / np.sqrt(len(win_rates_total)), 3)
                    counting_total = False
                if counting_total:
                    win_rates_total.append(int(line.split(' ')[1]))
                if counting_game:
                    win_rates_game.append(int(line.split(' ')[1]))
            # win_rate = round(float(lines[-1].strip().split(' ')[-1])*100)
            # solution = lines[-102].split(":")[-1].strip()
            solution = solution_mapping(solution)
            better = '*' if win_rate_g > old_win_rate[game] else ''
            equal = '=' if win_rate_g == old_win_rate[game] else ''
            if game > -1:
                # print(game_mapping[game].ljust(20, ' ') + str(win_rate_g) + "\t\t" + str(old_win_rate[game]) + "\t"
                print(str(game).ljust(20, ' ') + str(win_rate_g) + "\t\t" + str(old_win_rate[game]) + "\t"
                      + better + equal + "\t\t" + str(win_rate_total) + "\t\t\t" + solution)
                # if win_rate_g != old_win_rate[game]:
                #     print(stats.ttest_1samp(win_rates_game, old_win_rate[game]))
                print(win_rate_g_sterr, win_rate_total_sterr)
            else:
                change = int(f.split('_')[4].split('.txt')[0])
                print(("-1 (" + str(change + 1) + ")").ljust(20, ' ') + "\t\t\t\t\t" + str(
                    win_rate_total) + "\t\t\t" + solution)


def fitness_plots(only_files):
    print("Plotting fitness ...")
    plot_path = "fitness-plots-" + file_path
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    for f in only_files:
        fig = plt.figure()
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.grid(color='whitesmoke', ls='--')

        fitness, f_std, best, sota, sota_std = process_file_fitness(f)
        n = len(fitness)
        x = np.array(range(n))
        data = [[0 for _ in range(n)] for _ in range(6)]
        for i in range(n):
            data[int(fitness[i]/0.2)][i] = 1

        # ax.imshow(data, interpolation='bilinear', vmin=0.0, vmax=1.0, origin='lower', cmap='afmhot',
        #           extent=[0, n, -0.1, 1.1], aspect=n, zorder=-1)

        # from scipy.interpolate import make_interp_spline, BSpline
        # xnew = np.linspace(min(x), max(x), 300)
        # spl = make_interp_spline(x, fitness, k=1)  # type: BSpline
        # power_smooth = spl(xnew)
        # plt.plot(xnew, power_smooth, color='teal', ls='-', zorder=1)

        fitness=np.array(fitness)
        f_std=np.array(f_std)
        plt.fill_between(fitness-f_std, fitness+f_std, color='turquoise', zorder=1)
        plt.plot(fitness, color='paleturquoise', ls='-', zorder=1)

        sota=np.array(sota)
        sota_std=np.array(sota_std)
        plt.fill_between(sota-sota_std, sota+sota_std, color='teal', zorder=2)
        plt.plot(sota, color='teal', ls='-', zorder=2)

        # trend line
        # z = np.polyfit(x, fitness, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x), color='red', ls='--')

        # highlight best solutions
        for i in range(len(best)):
            if best[i] == 1:
                plt.axvline(i, color='black', alpha=0.5, zorder=3)
                # plt.scatter(i, fitness[i], color='black', alpha=1, zorder=3, s=100)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        plt.xlabel("NTBEA iteration", fontsize=28)
        plt.ylabel("N-tuple model value", fontsize=28)
        plt.ylim(-0.05, 1.05)
        plt.xlim(0, n)
        plt.tight_layout()
        plt.savefig(plot_path + "/" + f.split('.')[0] + ".png")

        # plt.show()
        plt.close(fig)


def solution_mapping(sol):
    """
    Receives string of solution in Java param order, turns it into paper param order:
    p.size, i.len, dd, o.cnt, elite, init, g.op, select, cross, mut, fit, d.w., f.skip, f.skipt, s.buffer, s.b.disc,
    mc.len, mc.rep
    :return: string of params in paper order
    """
    if sol != '':
        params = sol[1:-1].split(",")
        plist = []
        for p in params:
            plist.append(int(p.strip()))
        paper_sol = ["0" for _ in range(len(plist))]

        paper_sol[0] = str(plist[1])
        paper_sol[1] = str(plist[16])
        paper_sol[2] = str(plist[11])
        paper_sol[3] = str(plist[13])
        paper_sol[4] = str(plist[4])
        paper_sol[5] = str(plist[7])
        paper_sol[6] = str(plist[5])
        paper_sol[7] = str(plist[9])
        paper_sol[8] = str(plist[0])
        paper_sol[9] = str(plist[15])
        paper_sol[10] = str(plist[6])
        paper_sol[11] = str(plist[12])
        paper_sol[12] = str(plist[8])
        paper_sol[13] = str(plist[2])
        paper_sol[14] = str(plist[10])
        paper_sol[15] = str(plist[14])
        paper_sol[16] = str(plist[17])
        paper_sol[17] = str(plist[3])

        # return "[" + ", ".join(paper_sol) + "]"
        return " & ".join(paper_sol)
    else:
        return sol


def main():
    only_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    plot_1tuples_all_games(only_files)
    plot_2tuples_all_games(only_files)
    # plot_1tuples_per_game(only_files)
    # fitness_plots(only_files)
    # win_rate_analysis(only_files)

    # for f in only_files:
    #     process_file_fitness(f)


main()
