from src.config import *
from collections import Counter
from pprint import pprint


"""
Function to extract individual player features from the data. Fills the following arrays in the config file:
    - playerCS
    - playerCStotal
    - playerGold
    - playerGoldTotal
    - playerXP
    - playerXPtotal
    - playerDmg
    - playerDmgTotal
"""
def filterByPlayer(data):
    for g in data:  # for each game
        if g["gameId"] in all_bad_games:
            continue
        for p in g["participants"]:  # for each participant in the game

            # retrieve lane, role, timeline (stats per minute) and overall stats object
            thisLane = p["timeline"]["lane"]
            thisRole = p["timeline"]["role"]
            thisTimeline = p["timeline"]
            thisStats = p["stats"]
            playerLanes.append(thisLane)
            playerRoles.append(thisRole)

            # if thisLane not in lanes:
            #     print("lane: " + thisLane)
            # if thisLane == 'BOTTOM':
            #     if thisRole not in roles:
            #         pprint(p)

            try:  # get the features needed
                cs10 = thisTimeline.get("creepsPerMinDeltas").get("0-10")
                gold10 = thisTimeline.get("goldPerMinDeltas").get("0-10")
                xp10 = thisTimeline.get("xpPerMinDeltas").get("0-10")
                dmg10 = thisTimeline.get("damageTakenPerMinDeltas").get("0-10")
            except:  # if features not recorded for this player, mark them as 0
                cs10 = -1
                gold10 = -1
                xp10 = -1
                dmg10 = -1
                continue
            if cs10 is None:
                continue
            if gold10 is None:
                continue
            if xp10 is None:
                continue
            if dmg10 is None:
                continue

            # retrieve features from overall stats
            totalDmgTaken = thisStats.get("totalDamageTaken")
            totalMinionsKilled = thisStats.get("totalMinionsKilled")
            totalGoldEarned = thisStats.get("goldEarned")

            for l in range(len(lanes) - 1):  # for all but bottom lane, which contains two player roles
                if thisLane == lanes[l]:
                    # add in memory features from timeline for this particular role
                    playerCS[l].append(cs10)
                    playerGold[l].append(gold10)
                    playerXP[l].append(xp10)
                    playerDmg[l].append(dmg10)

                    # add in memory game and team ID for every player recorded
                    player_Gid[l].append(g["gameId"])
                    player_Tid[l].append(p["teamId"])

                    # add in memory features from overall stats for this particular role
                    playerCStotal[l].append(totalMinionsKilled)
                    playerGoldTotal[l].append(totalGoldEarned)
                    playerDmgTotal[l].append(totalDmgTaken)
                    break

            if thisLane == "BOTTOM":  # special case, 2 roles on bottom lane
                lane = len(lanes) - 1
                for r in range(len(roles)):
                    if thisRole == roles[r]:
                        # add in memory features from timeline for this particular role
                        playerCS[lane + r].append(cs10)
                        playerGold[lane + r].append(gold10)
                        playerXP[lane + r].append(xp10)
                        playerDmg[lane + r].append(dmg10)

                        # add in memory game and team ID for every player recorded
                        player_Gid[lane + r].append(g["gameId"])
                        player_Tid[lane + r].append(p["teamId"])

                        # add in memory features from overall stats for this particular role
                        playerCStotal[lane + r].append(totalMinionsKilled)
                        playerGoldTotal[lane + r].append(totalGoldEarned)
                        playerDmgTotal[lane + r].append(totalDmgTaken)
    return playerCS, playerCStotal, playerGold, playerGoldTotal, playerXP, playerXPtotal, playerDmg, playerDmgTotal, \
           player_Gid, player_Tid, playerLanes, playerRoles


"""
Function to extract overall team performance features from the data. Fills the following arrays from config:
    - teamCS
    - teamGold
    - teamDmg
    - win
"""
def getTeamData(data):
    for g in data:  # for each game
        if g["gameId"] in all_bad_games:
            continue
        team0 = g['teams'][0]  # get first team
        team1 = g['teams'][1]  # get second team

        # extract the other features as sums from the team members
        sumCS0, sumGold0, sumDmg0 = getSumsByTeamId(g, team0['teamId'])
        sumCS1, sumGold1, sumDmg1 = getSumsByTeamId(g, team1['teamId'])

        # add all features into memory
        teamCS.append(sumCS0)
        teamCS.append(sumCS1)
        teamGold.append(sumGold0)
        teamGold.append(sumGold1)
        teamDmg.append(sumDmg0)
        teamDmg.append(sumDmg1)
        win.append(team0['win'])
        win.append(team1['win'])
        teamId.append(team0['teamId'])
        teamId.append(team1['teamId'])
        team_GId.append(g['gameId'])
        team_GId.append(g['gameId'])
    return teamCS, teamGold, teamDmg, teamId, team_GId, win


"""
Function to retrieve sum of features for overall team by participants
Needs team ID and game stats object
"""
def getSumsByTeamId(g, teamId):
    sumCS = 0
    sumGold = 0
    sumDmg = 0
    for p in g["participants"]:
        thisStats = p["stats"]
        if p['teamId'] == teamId:
            sumCS += thisStats.get("totalMinionsKilled")
            sumGold += thisStats.get("goldEarned")
            sumDmg += thisStats.get("totalDamageTaken")
    return sumCS, sumGold, sumDmg


"""
Function to count the number of regions present in the data. Prints a dictionary mapping the region to its count.
"""
def countRegions(data):
    dataRegion = []
    for d in data:
        dataRegion.append(d["participantIdentities"][0]["player"]["platformId"])
    print(Counter(dataRegion))


"""
Function to reset global variables.
"""
def reset():
    global playerCS, playerCStotal, playerGold, playerGoldTotal, playerXP, playerXPtotal, playerDmg, playerDmgTotal,\
        player_Gid, player_Tid, playerLanes, playerRoles, teamCS, teamGold, teamDmg, teamId, team_GId, win
    playerCS = [[] for _ in range(no_players)]
    playerCStotal = [[] for _ in range(no_players)]
    playerGold = [[] for _ in range(no_players)]
    playerGoldTotal = [[] for _ in range(no_players)]
    playerXP = [[] for _ in range(no_players)]
    playerXPtotal = [[] for _ in range(no_players)]
    playerDmg = [[] for _ in range(no_players)]
    playerDmgTotal = [[] for _ in range(no_players)]
    player_Gid = [[] for _ in range(no_players)]
    player_Tid = [[] for _ in range(no_players)]
    playerLanes = []
    playerRoles = []
    teamCS = []
    teamGold = []
    teamDmg = []
    teamId = []
    team_GId = []
    win = []
