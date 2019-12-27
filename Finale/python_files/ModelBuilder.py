def rename_columns(dataframe):
    dataframe.rename(columns={'Winner': 'PlayerA',
                              'Loser': 'PlayerB',
                              'WRank': 'RankA',
                              'WRankAdded': 'RankFilledA',
                              'LRank': 'RankB',
                              'LRankAdded': 'RankFilledB',
                              'W1': '1A',
                              'L1': '1B',
                              'W2': '2A',
                              'L2': '2B',
                              'W3': '3A',
                              'L3': '3B',
                              'W4': '4A',
                              'L4': '4B',
                              'W5': '5A',
                              'L5': '5B',
                              'Wsets': 'setsA',
                              'Lsets': 'setsB',
                              'WPts': 'PtsA',
                              'LPts': 'PtsB',
                              'WPtsAdded': 'PtsFilledA',
                              'LPtsAdded': 'PtsFilledB',
                              'MaxW': 'MaxA',
                              'MaxWAdded': 'MaxFilledA',
                              'MaxL': 'MaxB',
                              'MaxLAdded': 'MaxFilledB',
                              'AvgW': 'AvgA',
                              'AvgWAdded': 'AvgFilledA',
                              'AvgL': 'AvgB',
                              'AvgLAdded': 'AvgFilledB',
                              }, inplace=True)
    dataframe.insert(0, column='Winner', value=0)


def get_inverted_dataFrame(dataframe, col_to_invert):
    print(col_to_invert)
    # dataframe = dataframe.reset_index(drop=True)
    # players = dataframe['PlayerA'].value_counts().subtract(
    #    dataframe['PlayerB'].value_counts(), fill_value=0).sort_values(ascending=False).index.tolist()[:25]

    # col_to_invert += players
    for f in col_to_invert:
        dataframe.iloc[1::2][[f + 'A', f + 'B']] = dataframe.iloc[1::2][[f + 'B', f + 'A']]
    dataframe.loc[1::2, 'Winner'] = 1
    return dataframe


def recalculate_comparisons(dataframe):
    dataframe['RankA>RankB'] = dataframe.apply(lambda x: int(x['RankA'] > x['RankB']), axis=1)
    dataframe['AvgA>AvgB'] = dataframe.apply(lambda x: int(x['AvgA'] > x['AvgB']), axis=1)
    dataframe['MaxA>MaxB'] = dataframe.apply(lambda x: int(x['MaxA'] > x['MaxB']), axis=1)

    dataframe['OpponentsWRatioA>OpponentsWRatioB'] = dataframe.apply(
        lambda x: int(x['OpponentsWRatioA'] > x['OpponentsWRatioB']), axis=1)

    dataframe['PlayedA>PlayedB'] = dataframe.apply(lambda x: int(x['PlayedA'] > x['PlayedB']), axis=1)
    dataframe['WonRatioA>WonRatioB'] = dataframe.apply(lambda x: int(x['WonRatioA'] > x['WonRatioB']), axis=1)

    dataframe['PlayedCourtA>PlayedCourtB'] = dataframe.apply(lambda x: int(x['PlayedCourtA'] > x['PlayedCourtB']),
                                                             axis=1)
    dataframe['WonRatioCourtA>WonRatioCourtB'] = dataframe.apply(
        lambda x: int(x['WonRatioCourtA'] > x['WonRatioCourtB']), axis=1)

    dataframe['FadigueTournGamesA>FadigueTournGamesB'] = dataframe.apply(
        lambda x: int(x['FadigueTournGamesA'] > x['FadigueTournGamesB']), axis=1)
    dataframe['FadigueTournSetsA>FadigueTournSetsB'] = dataframe.apply(
        lambda x: int(x['FadigueTournSetsA'] > x['FadigueTournSetsB']), axis=1)

    dataframe['PlayedVsSameHandedA>PlayedVsSameHandedB'] = dataframe.apply(
        lambda x: int(x['PlayedVsSameHandedA'] > x['PlayedVsSameHandedB']), axis=1)
    dataframe['WonRatioVsSameHandedA>WonRatioVsSameHandedB'] = dataframe.apply(
        lambda x: int(x['WonRatioVsSameHandedA'] > x['WonRatioVsSameHandedB']), axis=1)

    dataframe['5_gamesMeanA>5_gamesMeanB'] = dataframe.apply(
        lambda x: int(x['5_gamesMeanA'] > x['5_gamesMeanB']), axis=1)
    dataframe['5_setsMeanA>5_setsMeanB'] = dataframe.apply(
        lambda x: int(x['5_setsMeanA'] > x['5_setsMeanB']), axis=1)
