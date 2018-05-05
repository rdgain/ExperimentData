__author__ = 'dperez'

import numpy as np

def boldify(i):
    return '\\textbf{' + str(i) + '}'


def print_latex_table_results(data, num_games, num_alg, sig_p_value):

    print '\\begin{table*}[!t]'
    print '\\begin{center}\\resizebox{\\textwidth}{!}{'
    # print '\\begin{tabular}{|>{\\centering\\arraybackslash} m{1.5cm}|c|c|>{\\centering\\arraybackslash} m{1.7cm}|c|>{\\centering\\arraybackslash} m{1.7cm}|c|>{\\centering\\arraybackslash} m{1.7cm}|}'
    print '\\begin{tabular}{|c|c|c|c|c|c|}'
    print '\hline'
    # print '\\textbf{Game (Repetitions)}  & \\textbf{Algorithm} & \\textbf{Victories (\%)} &  \\textbf{Significantly better than ...} & \\textbf{Scores} &  \\textbf{Significantly better than ...} & \\textbf{Timesteps} &  \\textbf{Significantly better than ...} \\\\'
    print '\\textbf{Game}  & \\textbf{Algorithm} & \\textbf{Victories (\%)} &  \\textbf{Significantly better than ...} & \\textbf{Scores} &  \\textbf{Significantly better than ...}  \\\\'

    for gameidx in range(num_games):


        print '\hline'

        if num_alg > 1:
            #print '\multirow{3}{*}{\\textbf{' + GAME_NAMES[gameidx] + '}}'
            print '\multirow{' + str(num_alg - 1) + '}{*}{' + boldify(data[gameidx*num_alg][0]) + '}'
        else:
            print boldify(data[gameidx*num_alg][0]),


        for algidx in range(num_alg):

            lineidx =  gameidx*num_alg + algidx



            for i in range(len(data[lineidx])):
                strLine = data[lineidx][i]
                if i > 0:
                    if isinstance(strLine, basestring):
                        strLine = strLine.replace('_','\_');

                        if (i == 1):
                            if len(data[lineidx][3]) == (num_alg-1) or len(data[lineidx][5]) == (num_alg-1):

                                print '& ' + boldify(strLine),
                            else:
                                print '& ' + strLine,
                        else:
                            print '& ' + strLine,

                    else:
                        print '& ',
                        if len(strLine) == 0:
                            print '$\O$',
                        else:
                            for j in range(len(strLine)):
                                item = strLine[j]
                                if j < len(strLine)-1:
                                    print item + ',',
                                else:
                                    print item,



            print ' \\\\'



    print '\hline'
    print '\end{tabular}}'
    print '\caption{Percentage of victories and average of score achieved (plus standard error) in $%d$ different games. ' \
          'Fourth, sixth and eighth columns indicate the approaches that are significantly worse than that of the row, using the ' \
          'non-parametric Wilcoxon signed-rank test with p-value $<%.2f$.  Bold font for the ' \
          'algorithm that is significantly better than all the other $%d$ in either victories or score.}' % (num_games, sig_p_value, num_alg-1)
    print '\label{tab:weights}'
    print '\end{center}'
    print '\end{table*}'
    print ''



def latex_table_all_rankings(prettySumTable, num_games):


    # num_columns = num_games + 4  #rank, algorithm, points, avg_vict.
    # str_columns = '\\begin{tabular}{|'
    # for _ in range(num_columns):
    #     str_columns = str_columns + 'c|'
    # str_columns = str_columns + '}'

    col1 = '\\cellcolor{gray!45}'
    col2 = '\\cellcolor{gray!30}'
    col3 = '\\cellcolor{gray!15}'


    #'\\begin{tabular}{|>{\\centering\\arraybackslash} m{0.25cm}|>{\\centering\\arraybackslash} m{1.35cm}|>{\\centering\\arraybackslash} m{0.65cm}|>{\\centering\\arraybackslash} m{1.25cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|>{\\centering\\arraybackslash} m{0.455cm}|}'

    str_columns = '\\resizebox{\\textwidth}{!}{\\begin{tabular}{|c||>{\\centering\\arraybackslash} m{1.5cm}|>{\\centering\\arraybackslash} m{0.65cm}|>{\\centering\\arraybackslash} m{0.65cm}||'
    str_col_headers = '\\textbf{$\#$}  & \\textbf{Algorithm} & \\textbf{Points} &  \\textbf{Avg. Wins} '

    for i in range(num_games):
        str_col_headers = str_col_headers + ' & ' + boldify('G-' + str(i))
        str_columns += 'c|'

    str_col_headers = str_col_headers + ' \\\\'
    str_columns += '}'

    print '\\begin{table*}[!t]'
    print '\\begin{center}'
    print str_columns
    print '\hline'
    print str_col_headers
    print '\hline'

    for entry in prettySumTable:

        print '\hline'

        for i in range(len(entry)):

            datapiece = str(entry[i])
            if datapiece == '25' or i <= 1:
                datapiece = boldify(datapiece)

            if datapiece.__contains__('25') and i >= 4:
                datapiece = col3 + datapiece
            # if datapiece.__contains__('18') and i >= 4 :
            #     datapiece = col2 + datapiece
            # if datapiece.__contains__('15') and i >= 4 :
            #     datapiece = col3 + datapiece

            if i < len(entry) - 1:
                print datapiece + ' & ',
            else:
                print datapiece,

        print ' \\\\'

    print '\hline'
    print '\end{tabular}}'
    print '\caption{Rankings table for the compared algorithms across all games. In this order, the table shows the ' \
          'rank of the algorithms, their name, total points, average of victories and points achieved on each game, ' \
          'following the F1 Scoring system.}'
    print '\label{tab:weights}'
    print '\end{center}'
    print '\end{table*}'


def latex_table_games_rankings(prettyGameTable, list_games, game_names):
    for game_idx in list_games:
        latex_table_game_rankings(prettyGameTable[game_idx], game_names[game_idx])


def latex_table_game_rankings(prettyGameTable, game_name):

    print '\\begin{table*}[!t]'
    print '\\begin{center}'
    print '\\begin{tabular}{|c|c|c|c|c|c|}'
    print '\hline'
    print boldify(game_name) + ' & \\textbf{Algorithm} & \\textbf{Points} &  \\textbf{Winner \\%} & \\textbf{Avg. Score} & \\textbf{Avg. Timesteps} \\\\'

    for entry in prettyGameTable:

        print '\hline'

        for i in range(len(entry)):

            datapiece = str(entry[i])
            if datapiece == '25' or i <= 1:
                datapiece = boldify(datapiece)

            if i < len(entry) - 1:
                print datapiece + ' & ',
            else:
                print datapiece,

        print ' \\\\'

    print '\hline'
    print '\end{tabular}'
    print '\caption{Results for the game ' + game_name + ', showing rank, algorithm, points achieved, percentage of victories across all levels ' \
                                                       'and score and timesteps averages (standard error between parenthesis).}'
    print '\label{tab:weights}'
    print '\end{center}'
    print '\end{table*}'



def latex_table_array(dim1, dim2, table1, table2, caption, is_perc = True):

    num_rows = len(dim1)
    num_cols = len(dim2)

    str_columns = '\\begin{tabular}{|c|'
    col_header = ' '

    for x in range(num_cols):
        str_columns = str_columns + 'c|'
        col_header += ' & \\textbf{' + dim2[x] + '} '

    str_columns = str_columns + '}' + ' \\\\'
    col_header += ' \\\\'

    print '\\begin{table*}[!t]'
    print '\\begin{center}'
    print str_columns
    print '\hline'
    print col_header

    row_idx = 0
    for row in table1:

        print '\hline'

        print' \\textbf{' + dim1[row_idx] + '} ',

        for i in range(len(row)):

            str_val = '-'
            try:
               val = float(row[i])
               val2 = float(table2[row_idx][i])
               str_val = "%.2f (%.2f)" % (float(100*val), float(100*val2)) if is_perc else str(val)
            except ValueError:
               pass


            print ' & ' + str_val,

        print ' \\\\'
        row_idx += 1


    print '\hline'
    print '\end{tabular}'
    print '\caption{' + caption + '}'
    print '\label{tab:avgData}'
    print '\end{center}'
    print '\end{table*}'

