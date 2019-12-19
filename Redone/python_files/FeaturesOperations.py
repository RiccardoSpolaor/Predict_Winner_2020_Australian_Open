# Setto le statistiche tra giocatore 1 e 2 nel passato prima di quel match:
# Quante partite avevano giocato in precedenza giocatore A e giocatore B (OpponentsPlayed)
# Rapporto Vittorie/(Partite Giocate Tot) Tra i 2 giocatori (OpponentsWRatioA e OpponentsWRatioB)
import itertools
import numpy as np
import pandas as pd


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


def set_hands_statistics(dataframe):
    played_vs_same_handed_as_opponent_a = []
    played_vs_same_handed_as_opponent_b = []
    won_vs_same_handed_as_opponent_a = []
    won_vs_same_handed_as_opponent_b = []

    players_played_vs_right = {p: 0 for p in dataframe['Winner'].append(dataframe['Loser']).unique()}
    players_played_vs_left = {p: 0 for p in dataframe['Winner'].append(dataframe['Loser']).unique()}
    players_won_vs_right = {p: 0 for p in dataframe['Winner'].append(dataframe['Loser']).unique()}
    players_won_vs_left = {p: 0 for p in dataframe['Winner'].append(dataframe['Loser']).unique()}

    for item in dataframe.itertuples():
        if item.HandB == 0:
            played_vs_same_handed_as_opponent_a += [players_played_vs_right[item.Winner]]
            try:
                won_vs_same_handed_as_opponent_a += [
                    players_won_vs_right[item.Winner] / players_played_vs_right[item.Winner]]
            except:
                won_vs_same_handed_as_opponent_a += [0]

            players_played_vs_right[item.Winner] += 1
            players_won_vs_right[item.Winner] += 1

        else:
            played_vs_same_handed_as_opponent_a += [players_played_vs_left[item.Winner]]
            try:
                won_vs_same_handed_as_opponent_a += [
                    players_won_vs_left[item.Winner] / players_played_vs_left[item.Winner]]
            except:
                won_vs_same_handed_as_opponent_a += [0]
            players_played_vs_left[item.Winner] += 1
            players_won_vs_left[item.Winner] += 1

        if item.HandA == 0:
            played_vs_same_handed_as_opponent_b += [players_played_vs_right[item.Loser]]
            try:
                won_vs_same_handed_as_opponent_b += [
                    players_won_vs_right[item.Loser] / players_played_vs_right[item.Loser]]
            except:
                won_vs_same_handed_as_opponent_b += [0]
            players_played_vs_right[item.Loser] += 1
        else:
            played_vs_same_handed_as_opponent_b += [players_played_vs_left[item.Loser]]
            try:
                won_vs_same_handed_as_opponent_b += [
                    players_won_vs_left[item.Loser] / players_played_vs_left[item.Loser]]
            except:
                won_vs_same_handed_as_opponent_b += [0]
            players_played_vs_left[item.Loser] += 1

    dataframe['PlayedVsSameHandedA'] = played_vs_same_handed_as_opponent_a
    dataframe['PlayedVsSameHandedB'] = played_vs_same_handed_as_opponent_b
    dataframe['WonRatioVsSameHandedA'] = won_vs_same_handed_as_opponent_a
    dataframe['WonRatioVsSameHandedB'] = won_vs_same_handed_as_opponent_b

    dataframe['WonRatioVsSameHandedA>WonRatioVsSameHandedB'] = dataframe.apply(
        lambda x: int(x['WonRatioVsSameHandedA'] > x['WonRatioVsSameHandedB']), axis=1)


def set_players_hands(dataframe):
    def set_hand(players_hands):
        def get_hand(p):
            try:
                val = players_hands[p]
            except:
                val = np.nan
            return val

        dataframe['HandA'].fillna(dataframe[dataframe['HandA'].isna()]['Winner'].apply(get_hand), inplace=True)
        dataframe['HandB'].fillna(dataframe[dataframe['HandB'].isna()]['Loser'].apply(get_hand), inplace=True)

    # Dato che i nomi dei giocatori sono diversi da quelli nel dataframe di partenza procedo a riscriverli in modi
    # differenti e li confronto a tentativi con i nomi dei giocatori del dataframe originario.
    def give_new_name(p):
        splitted = p.split(' ')
        newstring = splitted[-1] + ' ' + ' '.join([n[0] + '.' for n in splitted[:-1]])
        return newstring.upper()

    def give_new_name_2(p):
        splitted = p.split(' ')
        if len(splitted) > 2:
            newstring = splitted[-2] + ' ' + splitted[-1] + ' ' + ' '.join([n[0] + '.' for n in splitted[:-2]])
            return newstring.upper()
        else:
            return np.nan

    def give_new_name_3(p):
        splitted = p.split(' ')
        newstring = splitted[-1] + ' ' + ''.join([n[0] + '.' for n in splitted[:-1]])
        return newstring.upper()

    def give_new_name_4(p):
        splitted = p.split(' ')
        if len(splitted) > 2:
            newstring = splitted[-2] + '-' + splitted[-1] + ' ' + ' '.join([n[0] + '.' for n in splitted[:-2]])
            return newstring.upper()
        else:
            return np.nan

    def give_new_name_5(p):
        splitted = p.split(' ')
        if len(splitted) > 2:
            newstring = splitted[-2] + ' ' + splitted[-1] + ' ' + ''.join([n[0] + '.' for n in splitted[:-2]])
            return newstring.upper()
        else:
            return np.nan

    def get_players_hands_dataframe(n):
        n = str(n)
        players_hands = extradf[['winner_name' + n, 'winner_hand']].drop_duplicates().append(
            extradf[['loser_name' + n, 'loser_hand']].rename(columns={'loser_name' + n: 'winner_name' + n,
                                                                      'loser_hand': 'winner_hand'}),
            ignore_index=True).drop_duplicates()
        players_hands['winner_hand'] = players_hands['winner_hand'].apply(lambda x: np.nan if x == 'U' else x)
        players_hands['winner_hand'].dropna(inplace=True, axis=0)

        return players_hands

    extradf = pd.read_csv('./datasets/ATP.csv', low_memory=False)

    # Esiste una riga che contiene un simbolo non valido per il nome del perdente, la elimino.
    extradf.drop(extradf[extradf['loser_name'] == ' '].index, inplace=True, axis=0)

    extradf['winner_name1'] = extradf['winner_name'].apply(give_new_name)
    extradf['loser_name1'] = extradf['loser_name'].apply(give_new_name)

    extradf['winner_name2'] = extradf['winner_name'].apply(give_new_name_2)
    extradf['loser_name2'] = extradf['loser_name'].apply(give_new_name_2)

    extradf['winner_name3'] = extradf['winner_name'].apply(give_new_name_3)
    extradf['loser_name3'] = extradf['loser_name'].apply(give_new_name_3)

    extradf['winner_name4'] = extradf['winner_name'].apply(give_new_name_4)
    extradf['loser_name4'] = extradf['loser_name'].apply(give_new_name_4)

    extradf['winner_name5'] = extradf['winner_name'].apply(give_new_name_5)
    extradf['loser_name5'] = extradf['loser_name'].apply(give_new_name_5)

    dataframe['HandA'] = np.nan
    dataframe['HandB'] = np.nan

    players_hands_df = get_players_hands_dataframe(1)
    players_hands_dict = {p: np.nan for p in players_hands_df['winner_name1']}
    for i in players_hands_df.itertuples():
        players_hands_dict[i.winner_name1] = i.winner_hand

    set_hand(players_hands_dict)

    print('Primo attempt ad exploiatare il dataframe:')
    print('HandA e HandB settati ancora a NaN:' + str(len(dataframe[dataframe['HandA'].isna()]['Winner'].append(
        dataframe[dataframe['HandB'].isna()]['Loser']).unique())) + '\n')

    players_hands_df = get_players_hands_dataframe(2)
    players_hands_dict = {p: np.nan for p in players_hands_df['winner_name2']}
    for i in players_hands_df.itertuples():
        players_hands_dict[i.winner_name2] = i.winner_hand

    set_hand(players_hands_dict)

    print('Secondo attempt ad exploiatare il dataframe:')
    print('HandA e HandB settati ancora a NaN:' + str(len(dataframe[dataframe['HandA'].isna()]['Winner'].append(
        dataframe[dataframe['HandB'].isna()]['Loser']).unique())) + '\n')

    players_hands_df = get_players_hands_dataframe(3)
    players_hands_dict = {p: np.nan for p in players_hands_df['winner_name3']}
    for i in players_hands_df.itertuples():
        players_hands_dict[i.winner_name3] = i.winner_hand

    set_hand(players_hands_dict)

    print('Terzo attempt ad exploiatare il dataframe:')
    print('HandA e HandB settati ancora a NaN:' + str(len(dataframe[dataframe['HandA'].isna()]['Winner'].append(
        dataframe[dataframe['HandB'].isna()]['Loser']).unique())) + '\n')

    players_hands_df = get_players_hands_dataframe(4)
    players_hands_dict = {p: np.nan for p in players_hands_df['winner_name4']}
    for i in players_hands_df.itertuples():
        players_hands_dict[i.winner_name4] = i.winner_hand

    set_hand(players_hands_dict)

    print('Quarto attempt ad exploiatare il dataframe:')
    print('HandA e HandB settati ancora a NaN:' + str(len(dataframe[dataframe['HandA'].isna()]['Winner'].append(
        dataframe[dataframe['HandB'].isna()]['Loser']).unique())) + '\n')

    players_hands_df = get_players_hands_dataframe(5)
    players_hands_dict = {p: np.nan for p in players_hands_df['winner_name5']}
    for i in players_hands_df.itertuples():
        players_hands_dict[i.winner_name5] = i.winner_hand

    set_hand(players_hands_dict)

    print('Quinto attempt ad exploiatare il dataframe:')
    print('HandA e HandB settati ancora a NaN:' + str(len(dataframe[dataframe['HandA'].isna()]['Winner'].append(
        dataframe[dataframe['HandB'].isna()]['Loser']).unique())) + '\n')

    players_hands_dict = {
        'ABDULLA M.': 'R',
        'AHOUDA A.': 'R',
        'AL GHAREEB M.': 'R',
        'AL MUTAWA J.': 'R',
        'AL-ALAWI S.K.': 'R',
        'ALAWADHI O.': 'R',
        'ALI MUTAWA J.M.': 'R',
        'ALTMAIER D.': 'R',
        'ALVAREZ E.': 'R',
        'ANCIC I.': 'R',
        'ANDERSON O.': 'R',
        'ANDREEV A.': 'R',
        'ARAGONE JC': 'R',
        'ARTUNEDO MARTINAVARRO A.': 'R',
        'ASCIONE A.': 'R',
        'AVIDZBA A.': 'R',
        'BACHELOT J.F': 'R',
        'BAGHDATIS M.': 'R',
        'BAHROUZYAN O.': 'R',
        'BASSO A.': 'L',
        'BAUTISTA R.': 'R',
        'BELLIER A.': 'L',
        'BENCHETRIT E.': 'R',
        'BENNETEAU A.': 'R',
        'BERRETTINI M.': 'R',
        'BOGAERTS R.': 'L',
        'BOGOMOLOV A.': 'R',
        'BOGOMOLOV JR. A.': 'R',
        'BOGOMOLOV JR.A.': 'R',
        'BONZI B.': 'R',
        'BROOKSBY J.': 'R',
        'CARUANA L.': 'R',
        'CELIKBILEK A.': 'R',
        'CERUNDOLO F.': 'R',
        'CERVANTES I.': 'R',
        'CHAKI R.': 'R',
        'CHEKOV P.': 'R',
        'CONDOR F.': 'R',
        'CORRIE E.': 'R',
        'DAILEY Z.': 'R',
        'DASNIERES DE VEIGY J.': 'L',
        'DAVIDOVICH FOKINA A.': 'R',
        'DAVLETSHIN V.': 'R',
        'DAVYDENKO P.': 'R',
        'DE HEART R.': 'L',
        'DEEN HESHAAM A.': 'R',
        'DEL BONIS F.': 'L',
        "DELL'ACQUA M.": 'R',
        'DEV VARMAN S.': 'R',
        'DONSKI A.': 'R',
        'DUBRIVNYY A.': 'R',
        'DUTRA DA SILVA R.': 'R',
        'ESTRELLA BURGOS V.': 'R',
        'FISH A.': 'R',
        'FLANAGAN I.': 'R',
        'FORNELL M.': 'R',
        'FRITZ T.': 'R',
        'GALAN D.E.': 'R',
        'GALLARDO M.': 'R',
        'GAO X.': 'R',
        'GASTON H.': 'L',
        'GOJO B.': 'R',
        'GOMEZ A.': 'R',
        'GOMEZ L.': 'L',
        'GORANSSON T.': 'L',
        'GRANOLLERS PUJOL G.': 'R',
        'GRANOLLERS-PUJOL M.': 'R',
        'GRAY A.': 'R',
        'GROMLEY C.': 'R',
        'GRUBER K.': 'R',
        'GUCCIONE A.': 'R',
        'GUZMAN J.': 'R',
        'HAJI A.': 'R',
        'HAJJI A.': 'R',
        'HANFMANN Y.': 'R',
        'HANK T.': 'L',
        'HANTSCHEK M.': 'R',
        'HARRIS L.': 'R',
        'HARSANYI P.': 'R',
        'HERBERT P-H.': 'R',
        'HERBERT P.H': 'R',
        'HERNANDEZ-FERNANDEZ J.': 'R',
        'HUESLER M.A.': 'L',
        'IDMBAREK Y.': 'R',
        'JANVIER M.': 'R',
        'JONES G.D.': 'R',
        'JUBB P.': 'R',
        'JUN W.': 'R',
        'KACHMAZOV A.': 'R',
        'KIM K': 'R',
        'KING K.': 'L',
        'KIRKIN E.': 'R',
        'KODERISCH C.': 'R',
        'KOHLSCHREIBER P..': 'R',
        'KOLAR Z.': 'R',
        'KORDA S.': 'R',
        'KORTELING S.': 'R',
        'KUCERA V.': 'R',
        'KUNITCIN I.': 'R',
        'KUTAC R.': 'R',
        'KUZNETSOV AL.': 'R',
        'KUZNETSOV AN.': 'R',
        'KWIATKOWSKI T.S.': 'R',
        'KYPSON P.': 'R',
        'LAZOV A.': 'L',
        'LENZ J.': 'R',
        'LESHEM E.': 'R',
        'LEVINE I.': 'R',
        'LIN J.M.': 'R',
        'LOPEZ VILLASENOR G.': 'R',
        'LOPEZ-JAEN M.A.': 'R',
        'LU Y.': 'R',
        'LUQUE D.': 'R',
        'MADEN Y.': 'R',
        'MARCH O.': 'R',
        'MARIN L.': 'R',
        'MARRAI M.': 'R',
        'MARTINEZ P.': 'R',
        'MATHIEU P.': 'R',
        'MATOS-GIL I.': 'R',
        'MEISTER N.': 'R',
        'MILOJEVIC N.': 'R',
        'MIYAO J.': 'R',
        'MOELLER M.': 'R',
        'MONTEIRO J.': 'R',
        'MORAING M.': 'L',
        'MORONI G.': 'R',
        'MOTT B.': 'R',
        'MULLER A.': 'R',
        'MUNAR J.': 'R',
        'MUNOZ DE LA NAVA D.': 'L',
        'NADER M.': 'R',
        'NAVA E.': 'R',
        'NAVARRO-PASTOR I.': 'R',
        'NEDOVYESOV O.': 'R',
        'NIKI T.': 'R',
        "O'BRIEN A.": 'R',
        "O'CONNELL C.": 'R',
        "O'NEAL J.": 'R',
        'OJEDA LARA R.': 'R',
        'OLASO G.': 'R',
        'OPELKA R.': 'R',
        'ORTEGA-OLMEDO R.': 'R',
        'PASHA N.': 'R',
        'PFEIFFER K.': 'R',
        'PIROS Z.': 'R',
        'PODLIPNIK H.': 'R',
        'POLMANS M.': 'R',
        'POPYRIN A.': 'R',
        'PRASHANTH V.': 'R',
        'PRPIC A.': 'R',
        'PURCELL M.': 'R',
        'QUERRY S.': 'R',
        'QUIGLEY E.': 'R',
        'QURESHI A.': 'R',
        'RAMOS-VINOLAS A.': 'L',
        'RASCON T.': 'R',
        'REIX C.': 'R',
        'REYES-VARELA M.A.': 'R',
        'RIBA-MADRID P.': 'R',
        'ROBREDO R.': 'R',
        'RODIONOV J.': 'L',
        'ROUMANE R.': 'R',
        'RUEVSKI P.': 'R',
        'SABRY S.': 'R',
        'SAFIULLIN R.': 'R',
        'SAKAMOTO P.': 'R',
        'SALVA B.': 'R',
        'SAMPER-MONTANA J.': 'R',
        'SANCHEZ DE LUNA J.A.': 'R',
        'SARMIENTO R.': 'R',
        'SCHUTTLER P.': 'R',
        'SCHWARTZMAN D.': 'R',
        'SERDARUSIC N.': 'R',
        'SHANE R.': 'R',
        'SI Y.M.': 'R',
        'SILVA D.': 'R',
        'SILVA F.': 'L',
        'SINNER J.': 'R',
        'SMETHURST D.': 'R',
        'SMITH A.': 'R',
        'STATHAM J.': 'R',
        'STATHAM R.': 'R',
        'STEPANEK M.': 'R',
        'SULTAN-KHALFAN A.': 'R',
        'SVAJDA Z.': 'R',
        'TAKAHASHI Y.': 'R',
        'TOPIC J.': 'R',
        'TRAVAGLIA S.': 'R',
        'TRUJILLO G.': 'R',
        'TRUSENDI W.': 'R',
        'TYURNEV E.': 'R',
        'VAISSE M.': 'R',
        'VAN DER DIUM A.': 'R',
        'VAN DER MEER N.': 'R',
        'VAN DER MERWE I.': 'R',
        'VATUTIN A.': 'R',
        'VERDASCO M.': 'R',
        'VICENTE M.': 'R',
        'VILOCA J.A.': 'R',
        'VIOLA MAT.': 'R',
        'WALTER L.': 'R',
        'WANG Y.': 'R',
        'WANG Y. JR': 'R',
        'WANG Y. JR.': 'R',
        'WANG Y.T.': 'R',
        'WARD A.': 'R',
        'WATANUKI Y.': 'R',
        'WILLMAN D.': 'R',
        'XU J.C.': 'R',
        'YOON Y.': 'R',
        'YOUZHNY A.': 'R',
        'ZAYED M. S.': 'R',
        'ZEKIC M.': 'R',
        'ZHANG Z.': 'R',
        'ZHANG ZE': 'R',
        'ZHANG ZE.': 'R',
        'ZHANG ZH.': 'R'
    }

    set_hand(players_hands_dict)

    print('Filling manuale dei valori mancanti:')
    print('HandA e HandB settati ancora a NaN:' + str(len(dataframe[dataframe['HandA'].isna()]['Winner'].append(
        dataframe[dataframe['HandB'].isna()]['Loser']).unique())) + '\n')
