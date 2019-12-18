# Setto le statistiche tra giocatore 1 e 2 nel passato prima di quel match:
# Quante partite avevano giocato in precedenza giocatore A e giocatore B (OpponentsPlayed)
# Rapporto Vittorie/(Partite Giocate Tot) Tra i 2 giocatori (OpponentsWRatioA e OpponentsWRatioB)
import itertools
import numpy as np


def set_player_a_vs_player_b_statistics(dataframe):
    opponents_played = []
    opponents_wratio_a = []
    opponents_wratio_b = []

    players_played = {tuple(e for e in sorted(pair)): 0 for pair in itertools.combinations(
        dataframe['Winner'].append(dataframe['Loser']).unique(), 2)}

    players_won = {pair: 0 for pair in
                   itertools.permutations(dataframe['Winner'].append(dataframe['Loser']).unique(), 2)}

    for item in dataframe.itertuples():

        opponents_played += [players_played[tuple(e for e in sorted([item.Winner, item.Loser]))]]

        try:
            opponents_wratio_a += [players_won[(item.Winner, item.Loser)] / players_played[
                tuple(e for e in sorted([item.Winner, item.Loser]))]]
        except:
            opponents_wratio_a += [0]

        try:
            opponents_wratio_b += [players_won[(item.Loser, item.Winner)] / players_played[
                tuple(e for e in sorted([item.Winner, item.Loser]))]]
        except:
            opponents_wratio_b += [0]

        players_played[tuple(e for e in sorted([item.Winner, item.Loser]))] += 1
        players_won[(item.Winner, item.Loser)] += 1

    dataframe['OpponentsPlayed'] = opponents_played
    dataframe['OpponentsWRatioA'] = opponents_wratio_a
    dataframe['OpponentsWRatioB'] = opponents_wratio_b
    dataframe['OpponentsWRatioA>OpponentsWRatioB'] = dataframe.apply(
        lambda x: int(x['OpponentsWRatioA'] > x['OpponentsWRatioB']), axis=1)


# Setto la stanchezza del giocatore in quel torneo (FadigueTournGames = quanti game ha giocato fin'ora un giocatore)
# (FadigueTournSets = quanti set ha giocato fin'ora un giocatore)
# Poi aggiungo una feature che indica se è più stanco in termini di game e di set A o B

def set_fatigue(dataframe):
    dataframe.insert(dataframe.columns.get_loc('Lsets') + 1, column='FadigueTournGamesA', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('FadigueTournGamesA') + 1, column='FadigueTournGamesB', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('FadigueTournGamesB') + 1, column='FadigueTournSetsA', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('FadigueTournSetsA') + 1, column='FadigueTournSetsB', value=np.nan)

    values = {'FA': [], 'FB': [], 'FsetsA': [], 'FsetsB': []}

    players_daily = {p: {'F': 0, 'Fsets': 0}
                     for p in dataframe['Winner'].append(dataframe['Loser']).unique()}

    for _, group in dataframe.groupby(['csvID', 'ATP'], sort=False):

        players_daily = {p: {'F': 0, 'Fsets': 0} for p in players_daily}

        for r in group.itertuples():

            for i in ['F', 'Fsets']:
                values[i + 'A'] += [players_daily[r.Winner][i]]
                values[i + 'B'] += [players_daily[r.Loser][i]]

            players_daily[r.Winner]['F'] += r.W1 + r.W2 + r.W3 + r.W4 + r.W5 + r.L1 + r.L2 + r.L3 + r.L4 + r.L5
            players_daily[r.Loser]['F'] += r.W1 + r.W2 + r.W3 + r.W4 + r.W5 + r.L1 + r.L2 + r.L3 + r.L4 + r.L5
            players_daily[r.Winner]['Fsets'] += r.Wsets + r.Lsets
            players_daily[r.Loser]['Fsets'] += r.Wsets + r.Lsets

    dataframe['FadigueTournGamesA'] = values['FA']
    dataframe['FadigueTournGamesB'] = values['FB']
    dataframe['FadigueTournSetsA'] = values['FsetsA']
    dataframe['FadigueTournSetsB'] = values['FsetsB']

    dataframe['FadigueTournGamesA>FadigueTournGamesB'] = dataframe.apply(
        lambda x: int(x['FadigueTournGamesA'] > x['FadigueTournGamesB']), axis=1)
    dataframe['FadigueTournSetsA>FadigueTournSetsB'] = dataframe.apply(
        lambda x: int(x['FadigueTournSetsA'] > x['FadigueTournSetsB']), axis=1)


# feature che indicano se il giocatore l'ultima volta ha fatto walkover o si è ritirato
def set_retired_walkover_last(dataframe):
    retired_lastA = []
    walkover_lastA = []
    retired_lastB = []
    walkover_lastB = []

    players_off = {p: {'Retired': 0, 'Walkover': 0}
                   for p in dataframe['Winner'].append(dataframe['Loser']).unique()}

    for r in dataframe.itertuples():
        retired_lastA += [players_off[r.Winner]['Retired']]
        retired_lastB += [players_off[r.Loser]['Retired']]

        walkover_lastA += [players_off[r.Winner]['Walkover']]
        walkover_lastB += [players_off[r.Loser]['Walkover']]

        players_off[r.Loser]['Walkover'] = r.Walkover
        players_off[r.Loser]['Retired'] = r.Retired
        players_off[r.Winner]['Walkover'] = 0
        players_off[r.Winner]['Retired'] = 0

    dataframe['WalkoverLastA'] = walkover_lastA
    dataframe['WalkoverLastB'] = walkover_lastB
    dataframe['RetiredLastA'] = retired_lastA
    dataframe['RetiredLastB'] = retired_lastB


def set_played_matches(dataframe):
    dataframe.insert(dataframe.columns.get_loc('Loser') + 1, column='PlayedA', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('PlayedA') + 1, column='PlayedB', value=np.nan)

    dataframe.insert(dataframe.columns.get_loc('PlayedB') + 1, column='WonRatioA', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('WonRatioA') + 1, column='WonRatioB', value=np.nan)

    played_a = []
    played_b = []
    won_a = []
    won_b = []

    players_played = {p: 0 for p in dataframe['Winner'].append(dataframe['Loser']).unique()}
    players_won = {p: 0 for p in dataframe['Winner'].append(dataframe['Loser']).unique()}

    for item in dataframe.itertuples():
        played_a += [players_played[item.Winner]]
        played_b += [players_played[item.Loser]]

        try:
            won_a += [players_won[item.Winner] / players_played[item.Winner]]
        except:
            won_a += [0]

        try:
            won_b += [players_won[item.Loser] / players_played[item.Loser]]
        except:
            won_b += [0]

        players_played[item.Loser] += 1
        players_played[item.Winner] += 1

        players_won[item.Winner] += 1

    dataframe['PlayedA'] = played_a
    dataframe['PlayedB'] = played_b
    dataframe['WonRatioA'] = won_a
    dataframe['WonRatioB'] = won_b

    dataframe['WonRatioA>WonRatioB'] = dataframe.apply(lambda x: int(x['WonRatioA'] > x['WonRatioB']), axis=1)


def set_played_matches_by_court(dataframe):
    dataframe.insert(dataframe.columns.get_loc('MaxL') + 1, column='PlayedCourtA', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('PlayedCourtA') + 1, column='PlayedCourtB', value=np.nan)

    dataframe.insert(dataframe.columns.get_loc('PlayedCourtB') + 1, column='WonRatioCourtA', value=np.nan)
    dataframe.insert(dataframe.columns.get_loc('WonRatioCourtA') + 1, column='WonRatioCourtB', value=np.nan)

    played_court_a = []
    played_court_b = []
    won_court_a = []
    won_court_b = []

    players_played = {p: {'Clay': 0, 'Hard': 0, 'Grass': 0, 'Carpet': 0} for p in
                      dataframe['Winner'].append(dataframe['Loser']).unique()}
    players_won = {p: {'Clay': 0, 'Hard': 0, 'Grass': 0, 'Carpet': 0} for p in
                   dataframe['Winner'].append(dataframe['Loser']).unique()}

    for item in dataframe.itertuples():

        if item.Hard != 0:
            court = 'Hard'
        elif item.Grass != 0:
            court = 'Grass'
        elif item.Clay != 0:
            court = 'Clay'
        elif item.Carpet != 0:
            court = 'Carpet'

        played_court_a += [players_played[item.Winner][court]]
        played_court_b += [players_played[item.Loser][court]]

        try:
            won_court_a += [players_won[item.Winner][court] / players_played[item.Winner][court]]
        except:
            won_court_a += [0]

        try:
            won_court_b += [players_won[item.Loser][court] / players_played[item.Loser][court]]
        except:
            won_court_b += [0]

        players_played[item.Winner][court] += 1
        players_played[item.Loser][court] += 1

        players_won[item.Winner][court] += 1

    dataframe['PlayedCourtA'] = played_court_a
    dataframe['PlayedCourtB'] = played_court_b
    dataframe['WonRatioCourtA'] = won_court_a
    dataframe['WonRatioCourtB'] = won_court_b

    dataframe['WonRatioCourtA>WonRatioCourtB'] = dataframe.apply(
        lambda x: int(x['WonRatioCourtA'] > x['WonRatioCourtB']), axis=1)


# Calcolo la media dei game vinti del 1o 2o 3o 4o 5o set per le ultime 10 partite per quel giocatore e i set vinti
# per le ultime 10 partite del giocatore

def set_players_last_5_statistics(dataframe):
    def get_player_last_stats(p, w):
        if len(p['1']) == 0:
            w['5_1Mean'] += [0]
            w['5_2Mean'] += [0]
            w['5_3Mean'] += [0]
            w['5_4Mean'] += [0]
            w['5_5Mean'] += [0]
            w['5_setsMean'] += [0]
        else:
            w['5_1Mean'] += [np.mean(p['1'])]
            w['5_2Mean'] += [np.mean(p['2'])]
            w['5_3Mean'] += [np.mean(p['3'])]
            w['5_4Mean'] += [np.mean(p['4'])]
            w['5_5Mean'] += [np.mean(p['5'])]
            w['5_setsMean'] += [np.mean(p['sets'])]

    w5 = {'5_1Mean': [],
          '5_2Mean': [],
          '5_3Mean': [],
          '5_4Mean': [],
          '5_5Mean': [],
          '5_setsMean': []}

    l5 = {'5_1Mean': [],
          '5_2Mean': [],
          '5_3Mean': [],
          '5_4Mean': [],
          '5_5Mean': [],
          '5_setsMean': []}

    players = {p: {'1': [], '2': [], '3': [], '4': [], '5': [], 'sets': []} for p in
               dataframe['Winner'].append(dataframe['Loser']).unique()}

    for r in dataframe.itertuples():
        get_player_last_stats(players[r.Winner], w5)

        get_player_last_stats(players[r.Loser], l5)

        players[r.Winner]['1'] = (players[r.Winner]['1'] + [r.W1])[-10:]
        players[r.Winner]['2'] = (players[r.Winner]['2'] + [r.W2])[-10:]
        players[r.Winner]['3'] = (players[r.Winner]['3'] + [r.W3])[-10:]
        players[r.Winner]['4'] = (players[r.Winner]['4'] + [r.W4])[-10:]
        players[r.Winner]['5'] = (players[r.Winner]['5'] + [r.W5])[-10:]
        players[r.Winner]['sets'] = (players[r.Winner]['sets'] + [r.Wsets])[-10:]

        players[r.Loser]['1'] = (players[r.Loser]['1'] + [r.L1])[-10:]
        players[r.Loser]['2'] = (players[r.Loser]['2'] + [r.L2])[-10:]
        players[r.Loser]['3'] = (players[r.Loser]['3'] + [r.L3])[-10:]
        players[r.Loser]['4'] = (players[r.Loser]['4'] + [r.L4])[-10:]
        players[r.Loser]['5'] = (players[r.Loser]['5'] + [r.L5])[-10:]
        players[r.Loser]['sets'] = (players[r.Loser]['sets'] + [r.Lsets])[-10:]

    dataframe['5_1MeanA'] = w5['5_1Mean']
    dataframe['5_2MeanA'] = w5['5_2Mean']
    dataframe['5_3MeanA'] = w5['5_3Mean']
    dataframe['5_4MeanA'] = w5['5_4Mean']
    dataframe['5_5MeanA'] = w5['5_5Mean']
    dataframe['5_setsMeanA'] = w5['5_setsMean']
    dataframe['5_1MeanB'] = l5['5_1Mean']
    dataframe['5_2MeanB'] = l5['5_2Mean']
    dataframe['5_3MeanB'] = l5['5_3Mean']
    dataframe['5_4MeanB'] = l5['5_4Mean']
    dataframe['5_5MeanB'] = l5['5_5Mean']
    dataframe['5_setsMeanB'] = l5['5_setsMean']
