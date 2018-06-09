lanes = ["JUNGLE", "TOP", "MIDDLE", "BOTTOM"]
roles = ["DUO_SUPPORT", "DUO_CARRY"]  # use only for 'BOTTOM' lane to differenciate the 2 players
players = ["Jungle", "Top", "Mid", "Support", "Carry"]
no_players = 5

"""
Arrays to store player data. 1000x5, each row is one match, each column is a player role.
"""
playerCS = [[] for _ in range(no_players)]
playerCStotal = [[] for _ in range(no_players)]

playerGold = [[] for _ in range(no_players)]
playerGoldTotal = [[] for _ in range(no_players)]

playerXP = [[] for _ in range(no_players)]
playerXPtotal = [[] for _ in range(no_players)]

playerDmg = [[] for _ in range(no_players)]
playerDmgTotal = [[] for _ in range(no_players)]

player_Gid = [[] for _ in range(no_players)]  # game ID
player_Tid = [[] for _ in range(no_players)]  # team ID (either 100 for blue or 200 for red)

playerLanes = []
playerRoles = []

"""
Arrays to store team data. 1000x1, each row is one match, each column is overall team value.
"""
teamCS = []
teamGold = []
teamDmg = []
teamId = []
team_GId = []
win = []  # Can be 'Win' or 'Fail'

""" 
Other data.
"""
bad_games = [[] for _ in range(no_players)]  # games in which there are not exactly 2 players for each role
all_bad_games = set()
