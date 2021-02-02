from SetUp import *
from PricingAndCalibration import Pricing


class BuildInputs(Pricing):

    def __init__(self, udl, matu):
        super(BuildInputs, self).__init__()

        self.udl = udl
        self.matu = matu
        self.df_params = pd.read_pickle(folder2 + '/Parameters_' + udl + '.pkl')
        self.df_volume = pd.read_pickle(folder1 + '/Execs_' + udl + '.pkl')
        self.df_udl = pd.read_pickle(folder1 + '/UDL_' + udl + '.pkl')
        # self.list_matu = sorted(list(self.df_params['ExpiDate'].unique()))
        self.df_volume = self.df_volume.loc[self.df_volume.MaturityDate == matu]
        self.df_params = self.df_params.loc[self.df_params.ExpiDate == matu].iloc[3:, :]
        #  eliminate the 3 first parameters to give allow for convergence

    def find_middle(self, a, b):  # in minutes of open market
        if (a == a) and (b == b):
            ti = self.timeline.index(b) - self.timeline.index(a)
            return self.timeline[self.timeline.index(a) + int(ti / 2)]
        else:
            return np.nan

    def even_index(self):
        date_ranges = []
        start = self.df_volume.index.min()
        non_empty_days = sorted(list(self.df_udl.index.unique()))
        for date in list(
                dict.fromkeys([elt.date() for elt in non_empty_days if ((elt >= start) and (elt <= self.matu))])):
            t1 = datetime.datetime.combine(date, self.opening_hours)
            t2 = datetime.datetime.combine(date, self.closing_hours)
            date_ranges.append(pd.DataFrame({"OrganizedDateTime": pd.date_range(t1, t2, freq='1Min').values}))
        agg = pd.concat(date_ranges, axis=0)
        # agg.index = agg["OrganizedDateTime"].values

        # we center the index of df_params on the middle of the cluster
        self.timeline = agg["OrganizedDateTime"].tolist()
        self.df_params.index = [self.find_middle(a, b) for a, b in
                                zip(self.df_params.StartTime, self.df_params.StartTime.shift(-1))]
        # self.df_params.index = self.df_params.index.map(lambda x: x.round('1min'))
        # self.df_params = self.df_params.iloc[1:, :]

        # we make sure that the index list of params is unique (not 2 calibration in the same minute)
        self.df_params = self.df_params.groupby(self.df_params.index).mean()

        # we reindex
        self.df_params = self.df_params.reindex(agg["OrganizedDateTime"].values)

        features = ['RefSpot', 'EWMA_ATF', 'EWMA_SMI', 'EWMA_CVX', 'EWMA_FwdRatio']
        for f in features:
            self.df_params[f] = self.df_params[f].interpolate(limit=60 * 8,
                                                              limit_area='inside')  # use 'time' option? + limit says that we won't extraoplate beyond 1 day

    def get_total_sensi(self):

        self.df_volume = pd.merge(self.df_volume, self.df_params, left_index=True, right_index=True, how='left')

        for f in ['EWMA_ATF', 'EWMA_SMI', 'EWMA_CVX', 'EWMA_FwdRatio']:
            self.df_volume[f + '_prec'] = self.df_volume[f].shift(1)

        for prec_or_not in ['', '_prec']:
            self.df_volume['vi' + prec_or_not], self.df_volume['delta' + prec_or_not], self.df_volume[
                'sensiATF' + prec_or_not], self.df_volume['sensiSMI' + prec_or_not], self.df_volume[
                'sensiCVX' + prec_or_not] = \
                self.df_volume.apply(lambda x: self.get_vol_and_sensi(x.PriceU, x.RefSpot, x.StrikePrice, x.TTM,
                                                                      x['EWMA_ATF' + prec_or_not],
                                                                      x['EWMA_SMI' + prec_or_not],
                                                                      x['EWMA_CVX' + prec_or_not],
                                                                      x['EWMA_FwdRatio' + prec_or_not], True), axis=1,
                                     result_type='expand')

            self.df_volume['Price' + prec_or_not] = self.df_volume.apply(
                lambda x: self.vanilla_pricer(x.PriceU, x.StrikePrice, x.TTM, 0,
                                              x['vi' + prec_or_not], x['EWMA_FwdRatio' + prec_or_not], x.PutOrCall),
                axis=1)

        self.df_volume['TotalSensiATF'] = self.df_volume.apply(
            lambda x: (self.vanilla_pricer(x.PriceU, x.StrikePrice, x.TTM, 0, x.vi + x.sensiATF, x.EWMA_FwdRatio,
                                           x.PutOrCall) - x.Price) / max(1 / 52, x.TTM) ** 0.5 * x.NumberOfContracts,
            axis=1)
        # it is actually weighted vega annualized
        self.df_volume['TotalSensiSMI'] = self.df_volume.apply(
            lambda x: (self.vanilla_pricer(x.PriceU, x.StrikePrice, x.TTM, 0, x.vi + x.sensiSMI, x.EWMA_FwdRatio,
                                           x.PutOrCall) - x.Price) * x.NumberOfContracts, axis=1)
        self.df_volume['TotalSensiFwdRatio'] = self.df_volume.apply(
            lambda x: (self.vanilla_pricer(x.PriceU * 1.01, x.StrikePrice, x.TTM, 0, x.vi + x.delta, x.EWMA_FwdRatio,
                                           x.PutOrCall) - x.Price) * x.NumberOfContracts, axis=1)

        # inorder to knnow if it isuy or sell trade, we look at the direction of the parameters (since trades are shifting the parameters)
        self.df_volume['TotalSignedSensiATF'] = self.df_volume['TotalSensiATF'] * (
                    self.df_volume['Price'] - self.df_volume['Price_prec'])
        self.df_volume['TotalSignedSensiSMI'] = self.df_volume['TotalSensiSMI'] * (
                    self.df_volume['Price'] - self.df_volume['Price_prec'])
        self.df_volume['TotalSignedSensiFwdRatio'] = self.df_volume['TotalSensiFwdRatio'] * (
                    self.df_volume['Price'] - self.df_volume['Price_prec'])

        self.df_volume['NumberOfTrades'] = self.df_volume['NumberOfTrades'].fillna(0)
        self.df_volume = self.df_volume[
            ['TotalSignedSensiATF', 'TotalSignedSensiSMI', 'TotalSignedSensiFwdRatio', 'TotalSensiATF', 'TotalSensiSMI',
             'TotalSensiFwdRatio', 'NumberOfTrades']]
        self.df_volume = self.df_volume.groupby(self.df_volume.index).sum()

        self.df_params = self.df_params[['EWMA_ATF', 'EWMA_SMI', 'EWMA_CVX', 'EWMA_FwdRatio']]
        self.df_params = self.df_params.groupby(self.df_params.index).mean()

    def merge(self):
        self.df = pd.merge(self.df_params, self.df_volume, left_index=True, right_index=True, how='left')
        self.df = pd.merge(self.df, self.df_udl[['PriceU', 'TradedVolume']], left_index=True, right_index=True,
                           how='left')

        self.df['PriceU'] = self.df['PriceU'].interpolate('time')  # should take nights into account

        for f in ['TradedVolume', 'TotalSensiATF', 'TotalSensiSMI', 'TotalSensiFwdRatio', 'TotalSignedSensiATF',
                  'TotalSignedSensiSMI', 'TotalSignedSensiFwdRatio', 'NumberOfTrades']:
            self.df[f] = self.df[f].fillna(0)

        self.df = self.df.dropna()
        # solve duplicate index:
        self.df = self.df.groupby(self.df.index).mean()

        self.df['MaturityDate'] = self.matu
