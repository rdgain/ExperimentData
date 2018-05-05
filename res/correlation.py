import pylab
from scipy.stats.stats import pearsonr
import numpy as np
import itertools

# file = "resultdata.txt"
file = "resultdatalevels.txt"
results = pylab.loadtxt(file, comments='*', delimiter=' ', usecols=range(9))

f3 = 2
feat3 = results[:,f3][:]

games = range(101)
games.remove(53)
games.remove(100)

alg = range(8)

alggames = list(itertools.product(games, alg))
uniquegames = set()
uniquealgs = set()

for g in alggames:
    resultsG = []
    for i in range(len(results)):
        if results[i][0] == g[0] and results[i][2] == g[1]:
            resultsG.append(results[i])
    resultsG = np.array(resultsG)

    # print resultsG

    for f1 in range(3, len(results[0])):
            # if f1 == f3:
            #     continue
        for f2 in range(f1+1, len(results[0])):
            if f1 == f2:
                continue
            try:
                feat1 = resultsG[:,f1][:]
                feat2 = resultsG[:,f2][:]

                r, p = pearsonr(feat1, feat2)
                # r, p = pearsonr(feat1, feat3)
                if (r > 0.5 or r < -0.5) and p < 0.05:
                    if f1 == 6 and f2 == 7:
                        uniquegames.add(g[0])
                        uniquealgs.add(g[1])
                        print g, f1, f2, r, p
            except:
                continue

print len(uniquegames), len(uniquealgs)
