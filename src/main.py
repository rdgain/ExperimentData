import json
from src.plot import *
from src.compute import *
from src.classification import *
from src.visualization import *
from pprint import pprint
from scipy.stats.stats import pearsonr


"""
Retrieve the data from files in a list, each game is an element in the list
"""
data = []
for i in range(1, 11):
    data.extend(json.load(open('../data/matches' + str(i) + '.json'))["matches"])
# print(len(data))
# pprint(data[0])  # print first game to see data available per game
# for g in data:  # print specific game with ID
#     if g["gameId"] == 2585567847:
#         pprint(g)
#         break


"""
Count regions
"""
# countRegions(data)

"""
Filter data per player and get per player data
"""
filterByPlayer(data)

"""
Get overall team data
"""
getTeamData(data)

"""
Remove bad games from data.
"""

# Get games in which there are not exactly 2 players
for i in range(no_players):
    count = Counter(player_Gid[i])
    for c in count.items():
        if c[1] != 2:
            bad_games[i].append(c[0])
all_bad_games = [y for x in bad_games for y in x]

# Get ambiguous games that have Bottom lane roles undefined, set as 'DUO' only
ambiguous_games = []
for i in team_GId:
    if i not in player_Gid[3] or i not in player_Gid[4]:
        ambiguous_games.append(i)
all_bad_games.extend(ambiguous_games)
all_bad_games = set(all_bad_games)

# Trim the data
trimmedData = []
for g in data:
    if g["gameId"] in all_bad_games:
        continue
    trimmedData.append(g)
data = trimmedData

# Reset and retrieve correct player data
reset()
playerCS, playerCStotal, playerGold, playerGoldTotal, playerXP, playerXPtotal, playerDmg, playerDmgTotal, \
player_Gid, player_Tid, playerLanes, playerRoles = filterByPlayer(data)
teamCS, teamGold, teamDmg, teamId, team_GId, win = getTeamData(data)


"""
Printing player role summaries to check for errors in data logged
"""
# print(len(data))
# print(Counter(playerLanes))
# print(Counter(playerRoles))

# --- Plotting zone

# plt.ion()  # Comment this out if not plotting anything.

"""
Plot player CS data
"""
# plotPlayerData(playerCS, 'creep score', 'cs', 'C1')
# plotPlayerData(playerCStotal, 'creep score', 'csTotal', 'C2')

"""
Plot player gold data
"""
# plotPlayerData(playerGold, 'gold earned', 'gold', 'C1')
# plotPlayerData(playerGoldTotal, 'gold earned', 'goldTotal', 'C2')

"""
Plot player damage data
"""
# plotPlayerData(playerDmg, 'damage taken', 'damage', 'C1')
# plotPlayerData(playerDmgTotal, 'damage taken', 'damageTotal', 'C2')

""" 
Plot player vs team data
"""
# plotPlayerTeamData(playerCS, teamCS, 'Creep Score', 'PvsT', None, win)
# plotPlayerTeamData(playerGold, teamCS, 'Gold', 'PvsT', 'C4', None)

# for i in range(no_players):
#     print(pearsonr(playerCS[i], teamCS))


"""
Classification
"""
# classify()

"""
Data visualisation
"""
# visuals()
