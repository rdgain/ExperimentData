import pylab
import os
import shutil
import numpy as np
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove

pop_size = [2, 10]#[2, 10, 10]#[30, 10, 90]  # [2, 10, 10]
ind_length = [8,14]#[8, 14, 10]#[30, 90, 10]  # [8, 14, 10]
init = [0, 2]
buffer = ['false', 'true']
alg = ['true']
games = range(102)
games.remove(73)
games.remove(83)
games = [53, 100]
levels = range(5)
rep = range(20)

errlist = set()
missing = []
emptylist = []
goodlist = []

counter = 0
for k in range(len(pop_size)):
    pop = pop_size[k]
    length = ind_length[k]
    for i in init:
        for b in buffer:
            for g in games:
                # for l in levels:
                #     emplist = []
                #     for r in rep:
                        config = str(length) + "_" + str(pop) + "_" + str(i) + "_" + str(b)
                        # config = str(length) + "_" + str(pop)
                        file = "rhea-journal/results/" + alg[0] + "_" + config + "/" + alg[0] + "_" + config + "_" + str(g) + ".txt"
                        # file = "rhea-journal/actfiles/act_" + alg[0] + "_" + config + "_" + str(g) + "_" + str(l) + "_" + str(r) + ".log"
                        # file = "rhea-journal/evofiles/evo_" + config + "_" + str(g) + "_" + str(l) + "_" + str(r) + ".log"
                        try:
                            if os.path.isfile(file):  # file exists
                                if os.stat(file).st_size == 0:
                                    # f = "Empty: " + config + "_" + str(g) + "_" + str(l) + "_" + str(r)
                                    f = "Empty: " + config + "_" + str(g)
                                    emptylist.append(f)
                                    # emplist.append(r)
                                    continue
                                # resultsGame = pylab.loadtxt(file, comments='*', delimiter=' ', usecols=range(2))
                                # resultsGame = np.genfromtxt(file, delimiter=' ', skip_footer=1)
                                # if resultsGame.shape[1] == 2:
                                goodlist.append(file)
                                # else:
                                #     errlist.append(f)
                            else:
                                f = "Missing: " + config + "_" + str(g)
                                # f = "Missing: " + config + "_" + str(g) + "_" + str(l) + "_" + str(r)
                                missing.append(f)
                                # emplist.append(r)
                        except:
                            f = "Error contents: " + config + "_" + str(g)
                            # f = "Error contents: " + config + "_" + str(g) + "_" + str(l) + "_" + str(r)
                            errlist.add(f)
                        counter += 1
                        # print counter, len(goodlist), len(errlist), len(emptylist), len(missing)

                    # check and replace all empty files
                    # for r in emplist:
                    #     if r == 0:
                    #         newR = 19
                    #     else:
                    #         newR = r-1
                    #     configFrom = str(length) + "_" + str(pop) + "_" + str(i) + "_" + str(b) + "_" + str(g) + "_" + str(l) + "_" + str(newR)
                    #     # configFrom = str(length) + "_" + str(pop) + "_" + str(g) + "_" + str(l) + "_" + str(newR)
                    #     configTo = str(length) + "_" + str(pop) + "_" + str(i) + "_" + str(b) + "_" + str(g) + "_" + str(l) + "_" + str(r)
                    #     # configTo = str(length) + "_" + str(pop) + "_" + str(g) + "_" + str(l) + "_" + str(r)
                    #     fileFrom = "rhea-journal/actfiles/act_" + alg[0] + "_" + configFrom + ".log"
                    #     # fileFrom = "rhea-journal/evofiles/evo_" + configFrom + ".log"
                    #     fileTo = "rhea-journal/actfiles/act_" + alg[0] + "_" + configTo + ".log"
                    #     # fileTo = "rhea-journal/evofiles/evo_" + configTo + ".log"
                    #     shutil.copy(fileFrom, fileTo)


print emptylist
print missing
print errlist
print len(goodlist), len(errlist), len(emptylist), len(missing)

for file in goodlist:
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file) as old_file:
            i = 0
            for line in old_file:
                splitline = line.split(" ")
                if len(splitline) == 1 or 'Controller' in line or 'Unexpected' in line:  # skip lines with 1 value or other errors
                    print "skip"
                    continue
                if 'Infinity' in line:
                    line.replace('Infinity', '1')
                    print "change inf"
                if 'NaN' in line:
                    line.replace('NaN', '0.00')
                    print "change NaN"
                if i % 2 == 1:
                    if len(splitline) < 14:
                        line = line[:-1] + " -1 -1 -1 -1\n"
                new_file.write(line)
                # print line
                i += 1
    # Remove original file
    remove(file)
    # Move new file
    move(abs_path, file)
