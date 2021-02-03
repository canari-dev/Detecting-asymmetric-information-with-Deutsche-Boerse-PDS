from SetUp import *


class Data():
    def __init__(self, udl):
        self.udl = udl
        self.df = pd.read_pickle(folder2 + '/Inputs_' + udl + '.pkl')
        # self.timeline = sorted(list(set(self.df.index.tolist())))
        self.list_expi = self.df.MaturityDate.unique()

        self.listdif1 = ['EWMA_ATF', 'EWMA_FwdRatio']
        self.listdif2 = ['TotalSignedSensiATF', 'TotalSignedSensiSMI', 'TotalSignedSensiFwdRatio']
        self.listdif3 = ['EWMA_SMI']  # do nothing
        self.listdif4 = ['TotalSensiATF', 'TotalSensiSMI', 'TotalSensiFwdRatio',
                         'NumberOfTrades']  # integrate over all matu
        self.listdif5 = ['TradedVolume']  # do nothing

    def find_nearby(self, row):
        res = [np.nan, np.nan, np.nan, np.nan]
        lst = [(time_between(row.name, elt), elt) for elt in self.list_expi if not np.isnan(row[elt])]
        if len(lst) > 0:
            res[0] = lst[0][0]
            res[1] = lst[0][1]
        if (len(lst) > 1) and (lst[1][0] < 60 * 60 * 24 * 30):  # dont use as nearby if more than 30days away
            res[2] = lst[1][0]
            res[3] = lst[1][1]
        return res

    def vol_fwd(self, row, f, rank):
        nb = row['nearby_date' + str(rank)]
        if nb in self.list_expi:
            t2 = time_between(row.name, f)
            var2 = t2 * row[f] ** 2
            t1 = row['nearby_TTM' + str(rank)]
            var1 = t1 * row[nb] ** 2

            if (t1 == t2) or (t2 / t1 - 1 < .5) or (var2 < 2 * var1):
                # if we can't compute a fwd vol because this is the nearby maturity or
                # or we want the fwd time to be greater than 150% of the time to nearby to avoid levying errors too much
                # or we in a dangerous case of very steep termstructure
                # then we return the vol itself
                return (row[f], 0)
            else:
                return (((var2 - var1) / (t2 - t1)) ** 0.5, 1)
        else:
            return (np.nan, 0)

    def get_ratio(self, row, f, rank):
        nb = row['nearby_date' + str(rank)]
        if nb in self.list_expi:
            if f == row['nearby_date' + str(rank)]:
                return (row[f], 0)
            else:
                return (row[f] / row[row['nearby_date' + str(rank)]], 1)
        else:
            return (np.nan, 0)

    def integrate(self, row, way, matu, stopat=-2):
        if matu in ['nearby_date0', 'nearby_date1']:
            matu = row[matu]
            if pd.isnull(matu):
                return 0
        if matu == 0:
            pos = 0
        else:
            pos = row.index.tolist().index(matu)

        if way == 'from':
            lst = [elt for elt in row[pos:stopat].values if
                   not np.isnan(elt)]  # stop at -2 if the last 2 columns are nearby dates
            return sum(lst)
        else:
            lst = [elt for elt in row[:pos + 1].values if not np.isnan(elt)]
            return sum(lst)

    def differentiate_matu(self):
        # !!!! passer en working days

        self.dfmultiindex = self.df[self.listdif1 + self.listdif2 + self.listdif3 + self.listdif4].set_index(
            [self.df.index, self.df.MaturityDate])
        self.df_pivot = self.dfmultiindex.unstack(-1)

        for f in self.listdif1:
            print(f)
            idx = pd.IndexSlice
            self.dft = self.df_pivot.loc[:, idx[f, :]]
            self.list_expi = [elt[1] for elt in self.dft.columns]
            self.dft.columns = self.list_expi
            self.list_expi = sorted(self.list_expi)
            self.dft = self.dft[self.list_expi]

            self.dft[['nearby_TTM0', 'nearby_date0', 'nearby_TTM1', 'nearby_date1']] = \
                self.dft.apply(lambda x: self.find_nearby(x), axis='columns', result_type='expand')

            for rank, lib in [(0, ''), (1, '2nd')]:
                self.df_pivot[('nearby_date' + str(rank), 'allexpi')] = self.dft['nearby_date' + str(rank)]

                for mt in self.list_expi:  # [pd.Timestamp('2019-12-20 15:30:00')]: #self.list_expi:
                    if f == 'EWMA_ATF':
                        df = self.dft.apply(lambda x: self.vol_fwd(x, mt, rank), axis='columns', result_type='expand')
                    if f == 'EWMA_FwdRatio':
                        df = self.dft.apply(lambda x: self.get_ratio(x, mt, rank), axis='columns', result_type='expand')

                    self.df_pivot[(f + lib, mt)] = df[0]
                    self.df_pivot[(f + lib + "-type", mt)] = df[1]

        for f in self.listdif2:
            print(f)

            # We create dft with f-related columns and the nearby dates
            idx = pd.IndexSlice
            self.dft = self.df_pivot.loc[:, idx[f, :]]
            self.dft = pd.concat(
                [self.dft, self.df_pivot.loc[:, [('nearby_date0', 'allexpi'), ('nearby_date1', 'allexpi')]]], axis=1,
                join="inner")
            # end then rename columns to keep only relevant level
            self.dft.columns = [elt[1] if elt[0][:6] != 'nearby' else elt[0] for elt in self.dft.columns]

            for mt in self.list_expi:
                self.df_pivot[(f + 'nodif', mt)] = self.dft.apply(lambda x: self.integrate(x, 'from', mt),
                                                                  axis='columns')
                self.df_pivot[(f + 'nearby0', mt)] = self.df_pivot[(f + 'nodif', mt)] - self.dft.apply(
                    lambda x: self.integrate(x, 'upto', 'nearby_date0'), axis='columns')
                self.df_pivot[(f + 'nearby1', mt)] = self.df_pivot[(f + 'nodif', mt)] - self.dft.apply(
                    lambda x: self.integrate(x, 'upto', 'nearby_date1'), axis='columns')
                self.df_pivot[(f + 'nearby0-date', mt)] = self.dft['nearby_date0']
                self.df_pivot[(f + 'nearby1-date', mt)] = self.dft['nearby_date1']

        for f in self.listdif3:
            print(f)
            # do nothing

        # self.listdif4
        for f in self.listdif4:
            print(f)
            idx = pd.IndexSlice
            self.dft = self.df_pivot.loc[:, idx[f, :]]
            self.dft.columns = [elt[1] for elt in self.dft.columns]
            self.df_pivot[f, 'allexpi'] = self.dft.apply(lambda x: self.integrate(x, 'from', 0, 1000), axis='columns')

        # self.listdif5
        newname = [(elt, 'allexpi') for elt in self.listdif5]
        self.dft = self.df[self.listdif5]
        self.dft.columns = newname
        self.dft = self.dft[~self.dft.index.duplicated(keep='first')]
        self.df_pivot = pd.concat([self.df_pivot, self.dft], axis=1, join="inner")

    def choose_sensi(self, x, cl, refcl):
        if x[refcl + '-type'] == 0:
            return x['dt-' + cl + 'nodif'], x[cl + 'nodif']
        elif x[refcl + '-nearby'] == x[cl + 'nearby0-date']:
            return x['dt-' + cl + 'nearby0'], x[cl + 'nearby0']
        elif x[refcl + '-nearby'] == x[cl + 'nearby1-date']:
            return x['dt-' + cl + 'nearby0'], x[cl + 'nearby0']
        else:
            return np.nan, np.nan

    def differentiate_time(self, st, lt, Ylag):
        print('differentiate_time')
        st_td = datetime.timedelta(days=st)
        lt_td = datetime.timedelta(days=lt)
        st_min = int(st * 60 * 8.5)
        lt_min = int(lt * 60 * 8.5)
        Ylag_min = int(Ylag * 60 * 8.5)

        # here we need to decide if we want NumberOfTrades summed up or for each matu
        for field in self.listdif4:
            self.df_pivot.drop(
                [elt for elt in self.df_pivot.columns if ((elt[0] == field) and (elt[1] != 'allexpi'))], axis=1,
                inplace=True)

        self.X = pd.DataFrame()
        for expi in self.list_expi:
            print(expi)

            self.listcol = [elt for elt in self.df_pivot.columns if elt[1] in [expi, 'allexpi']]

            self.dfu = self.df_pivot[self.listcol]
            self.listcol = [elt[0] for elt in self.listcol]
            self.dfu.columns = self.listcol

            self.Xt = pd.DataFrame()

            for cl in self.listdif1:
                print(cl)
                ncol = [cl, cl + '-nearby', cl + '-type']

                self.dftj = pd.DataFrame(columns=ncol)

                dft1 = self.dfu[[cl, 'nearby_date0', cl + '-type']]
                dft1.columns = ncol
                dft2 = self.dfu[[cl + '2nd', 'nearby_date1', cl + '2nd' + '-type']]
                dft2.columns = ncol
                self.dft = dft1.append(dft2)
                self.dft = self.dft.dropna(subset=[cl])

                # if type indicates that no differentiation has been done (type==0) then there is no nearby -> 1st jan 2000
                self.dft[cl + '-nearby'] = self.dft.apply(
                    lambda x: pd.Timestamp('2000-01-01 00:00:00') if x[cl + '-type'] == 0 else x[cl + '-nearby'],
                    axis=1)

                self.list_nearby = [elt for elt in list(set(self.dft[cl + '-nearby'].tolist())) if elt is not None]

                for nb in self.list_nearby:
                    self.dfti = self.dft.loc[self.dft[cl + '-nearby'] == nb]
                    self.dfti = self.dfti.groupby(
                        self.dfti.index).first()  # I don't see why there could be more than one line for the same time
                    self.dfti = self.dfti.sort_index()

                    # apply ewma to same nearby matu:
                    dfst = self.dfti[cl].ewm(span=st_min, min_periods=st_min).mean()
                    if st == 0:
                        self.dfti['dt-' + cl] = (self.dfti[cl] / self.dfti[cl].ewm(span=lt_min, min_periods=int(
                            lt_min / 2)).mean()) - 1
                    else:
                        self.dfti['dt-' + cl] = dfst / self.dfti[cl].ewm(span=lt_min,
                                                                         min_periods=int(lt_min / 2)).mean() - 1

                    # #here we prefer to calculate ewm using the actual date because it is good to create a gap in between days, espacially for diyield when there is an ex-date
                    # dfst = self.dfti[cl].ewm(halflife=st_td, times=self.dfti.index)).mean()
                    # if st==0:
                    #     self.dfti['dt-' + cl] = self.dfti[cl] / self.dfti[cl].ewm(halflife=lt_td, times=self.dfti.index).mean() - 1
                    # else:
                    #     self.dfti['dt-' + cl] = dfst / self.dfti[cl].ewm(halflife=lt_td, times=self.dfti.index).mean() - 1
                    #

                    self.dfti['Y-' + cl] = dfst.shift(-Ylag_min) / dfst - 1
                    # maybe add a protection against holes in the index to make sure shift(-Ylag_min) does take us approx Ylag days ahead
                    # self.dfti['validY'] = (dfst.shift(-Ylag_min).index - dfst.index).totalseconds() / (60 * 60 * 8.5)
                    # print('Before Y verif : ' + str(self.dfti.shape))
                    # self.dfti = self.dfti.loc[(self.dfti.validY < Ylag + 3)]  # 2 for a weekend, one for leeway
                    # print('After Y verif : ' + str(self.dfti.shape))

                    self.dftj = self.dftj.append(self.dfti)

                if self.dftj.shape[0] > 0:
                    # if self.udl!=ref: #we keep everything for the ref to make sure we wont loose lines when we intersect index
                    self.dftj = self.dftj.sort_values(
                        by=cl + '-nearby')  # so than when we take first(), the vol.ewm(st)/vol.ewm(lt) calculated with the shortest nearby will be kept
                    self.dftj = self.dftj.groupby(self.dftj.index).first()
                    # NB : it is important to do it only at this stage to make sure that the transition between the
                    # nearby maturities is done smoothly and time differentiation won't mix fwd vols with different nearby maturity
                    self.dftj = self.dftj.sort_index()

                self.Xt = pd.concat([self.Xt, self.dftj[[cl, 'dt-' + cl, cl + '-nearby', 'Y-' + cl, cl + '-type']]],
                                    axis=1)

            for cl in self.listdif2:
                print(cl)
                if cl == 'TotalSignedSensiFwdRatio':
                    refcl = 'EWMA_FwdRatio'
                else:
                    refcl = 'EWMA_ATF'

                unsigned_cl = cl.replace("Signed", "")

                self.dft = self.dfu[
                    [refcl, cl + 'nodif', cl + 'nearby0', cl + 'nearby1', cl + 'nearby0-date', cl + 'nearby1-date',
                     unsigned_cl]]
                self.dft = self.dft.dropna(subset=[refcl])  # refcl is only here in order to reduce size

                for f in [cl + 'nodif', cl + 'nearby0', cl + 'nearby1']:

                    if st == 0:
                        self.dft['dt-' + f] = (
                                self.dft[f] / self.dft[f].ewm(span=lt_min, min_periods=int(lt_min / 2)).mean())
                    else:
                        self.dft['dt-' + f] = (
                                self.dft[f].ewm(span=st_min, min_periods=st_min).mean() / self.dft[unsigned_cl].ewm(
                            span=lt_min,
                            min_periods=int(lt_min / 2)).mean())  # mean is already 0 so no need for a -1

                    #     if st == 0:
                    #         self.dft['dt-' + f] = (self.dft[f] / self.dft[f].ewm(halflife=lt_td, times=self.dfti.index).mean()) - 1
                    #     else:
                    #         self.dft['dt-' + f] = (self.dft[f].ewm(halflife=st_td, times=self.dft.index).mean() / self.dft[f].ewm(halflife=lt_td, times=self.dft.index).mean()) - 1

                self.dft = pd.concat([self.Xt[[refcl + '-nearby', refcl + '-type']], self.dft],
                                     axis=1)  # add nearby chosen for refcl to pick the one which matches among ['dt-'+cl + 'nodif', 'dt-'+cl + 'nearby0', 'dt-'+cl + 'nearby1'
                self.dft_chosen = self.dft.apply(lambda x: self.choose_sensi(x, cl, refcl), axis='columns',
                                                 result_type='expand')
                self.dft_chosen.columns = ['dt-' + cl, cl]

                self.Xt = pd.concat([self.Xt, self.dft_chosen], axis=1)

            for cl in self.listdif3 + self.listdif4 + self.listdif5:
                print(cl)

                self.dft = self.dfu[[refcl, cl]]  # we include refcl just to reduce size hereafter
                self.dft = self.dft.dropna(subset=[refcl])  # in order to reduce size

                if st == 0:
                    self.dft['dt-' + cl] = (self.dft[cl] / self.dft[cl].ewm(span=lt_min,
                                                                            min_periods=int(lt_min / 2)).mean()) - 1
                else:
                    self.dft['dt-' + cl] = (self.dft[cl].ewm(span=st_min, min_periods=st_min).mean() / self.dft[cl].ewm(
                        span=lt_min, min_periods=int(lt_min / 2)).mean()) - 1
                # is it better to follow the open time or the real time? when we get 30min interval calibration maybe revert to open time...

                # if st == 0:
                #     self.dft['dt-' + cl] = (self.dft[cl] / self.dft[cl].ewm(halflife=lt_td, times=self.dft.index).mean()) - 1
                # else:
                #     self.dft['dt-' + cl] = (self.dft[cl].ewm(halflife=st_td, times=self.dft.index).mean() / self.dft[cl].ewm(halflife=lt_td, times=self.dft.index).mean()) - 1

                self.Xt = pd.concat([self.Xt, self.dft[['dt-' + cl, cl]]], axis=1)

            self.Xt['Matu'] = expi
            self.X = self.X.append(self.Xt)

        # self.X = self.X.dropna()
        self.X = self.X.set_index([self.X.index, self.X.Matu])
        del self.X['Matu']

    def filter(self, TTM=1, type=1):
        # TTM
        self.X['TTM'] = self.X.index.get_level_values(1) - self.X.index.get_level_values(0)
        self.X['TTM'] = self.X['TTM'].apply(lambda x: x.total_seconds() / 60 / 60 / 24 / 365)
        self.X = self.X.loc[self.X.TTM < TTM]
        del self.X['TTM']

        if type == 1:
            for cl in [elt + '-type' for elt in self.listdif1]:
                self.X = self.X.loc[self.X[cl] != 0]
        if type == 2:
            for cl in [elt + '-type' for elt in self.listdif1]:
                self.X = self.X.loc[self.X[cl] != 1]

        rc = [elt for elt in self.X.columns if elt[-5:] != '-type']
        self.X = self.X[rc]

    def difref(self, x, ref):
        if x * ref <= 0:
            return x
        elif abs(ref) > 1:
            return x / ref
        else:
            return x

    def differentiate_refindex(self, XRef, exclude):
        value_cols = [elt for elt in self.X.columns if elt != 'udl']
        for takeout in exclude + ['nearby']:
            value_cols = [elt for elt in value_cols if (takeout not in elt)]

        idx = np.intersect1d(self.X.index, XRef.index)
        self.X = self.X.loc[idx]
        XRef = XRef.loc[idx]
        vecF = np.vectorize(self.difref)
        dfe = pd.DataFrame(vecF(self.X[value_cols], XRef[value_cols]))
        dfe.index = idx
        dfe.columns = value_cols
        self.X[value_cols] = dfe

    def Cat_Y(self, y, threshold):
        if y >= threshold:
            return (1)
        elif y <= -1 * threshold:
            return (-1)
        else:
            return (0)

    def normalize(self, cap):
        value_cols = [elt for elt in self.X.columns if 'nearby' not in elt]
        self.X[value_cols] = self.X[value_cols] - self.X[value_cols].mean()
        self.X[value_cols] = self.X[value_cols] / self.X[value_cols].std()
        for f in value_cols:
            self.X[f] = self.X[f].apply(lambda x: np.nan if np.isnan(x) else min(cap, max(x, -1 * cap)))

        Y_cols = [elt for elt in value_cols if elt[:2] == 'Y-']
        for f in Y_cols:
            self.X[f] = self.X[f].apply(lambda x: self.Cat_Y(x, cap))
