from SetUp import *
import matplotlib.dates as mdates


class Graph():
    def __init__(self, udl='DAI'):
        self.udl = udl
        try:
            if udl != '':
                self.dfp = pd.read_pickle(folder2 + '/Parameters_' + udl + '.pkl')
                self.dfi = pd.read_pickle(folder2 + '/Inputs_' + udl + '.pkl')
                self.dfx = pd.read_pickle(folder3 + '/X_' + udl + '.pkl')
                self.dfxtd = pd.read_pickle(folder3 + '/Xtd_' + udl + '.pkl')
                self.dfxy = pd.read_pickle(folder3 + '/XY_' + udl + '.pkl')
            else:
                self.dfxy = pd.read_pickle(folder3 + '/XY_all_stocks -st 2-lt 20.pkl')
        except:
            pass

    def graph_params(self, year=2020, month=9):
        expi1 = pd.Timestamp(str(year) + "-{:02d}".format(month) + "-15 15:30:00")
        listexpi = [expi1 + pd.Timedelta(i, unit='d') for i in range(7)]
        self.dfpt = self.dfp.loc[self.dfp.ExpiDate.isin(listexpi)]
        mask = (self.dfpt['StartTime'].dt.year == year)
        self.dfpt = self.dfpt[mask]

        self.dfpt['divyield'] = (1 - self.dfpt['FwdRatio']) * 100
        self.dfpt['EWMA_divyield'] = (1 - self.dfpt['EWMA_FwdRatio']) * 100
        self.dfpt['MSE (bps)'] = self.dfpt['Error']
        self.dfpt.index = self.dfpt.StartTime

        plt.close()
        self.dfpt[['EWMA_ATF', 'EWMA_SMI', 'EWMA_CVX', 'EWMA_divyield', 'MSE (bps)']].plot(
            secondary_y=['EWMA_ATF', 'EWMA_CVX'],
            title="Parameters for " + self.udl + " " + str(month) + "/" + str(year) + " maturity")
        plt.show()

    all_month = [i + 1 for i in range(12)]

    def graph_inputs(self, year=2020, expi_month=all_month, graph_month=all_month, field='EWMA_ATF'):
        self.l = list(dict.fromkeys(self.dfi.MaturityDate.tolist()))

        plt.close()
        for expi in [elt for elt in self.l if (elt.year == year) and (elt.month in expi_month)]:
            self.dfit = self.dfi.loc[self.dfi.MaturityDate == expi]

            mask1 = self.dfit.index.year == year
            self.dfit = self.dfit[mask1]

            self.dfit['month'] = self.dfit.index.month
            mask2 = self.dfit.month.isin(graph_month)
            self.dfit = self.dfit[mask2]

            ax = self.dfit[field].plot(title=self.udl + " : " + field + " for all maturities of " + str(year),
                                       grid=True, label=expi, legend=True)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %d'))
        plt.show()

    def graph_inputs_rg(self, fields=['EWMA_ATF', 'TotalSignedSensiATF']):
        plt.close()
        for rg in self.rg:
            self.dfit = self.dfi.loc[self.dfi.MaturityDate == rg[0][1]]

            lst_times = [elt[0] for elt in rg]
            lst_tim_ext5D = [elt for elt in self.dfit.index if
                             ((elt > lst_times[-1]) and (elt < lst_times[-1] + pd.Timedelta(7, unit='D')))]
            lst_tim_prev5D = [elt for elt in self.dfit.index if
                              ((elt < lst_times[0]) and (elt > lst_times[0] - pd.Timedelta(7, unit='D')))]
            lst_t = lst_tim_prev5D + lst_times + lst_tim_ext5D
            self.dfit = self.dfit.loc[lst_t, :]
            secondary = [elt for elt in fields if (('EWMA_ATF' in elt) or ('EWMA_CVX' in elt))]
            ax = self.dfit[fields].plot(title=self.udl + ", Matu : " + str(rg[0][1]), secondary_y=secondary, grid=True,
                                        legend=True)
            plt.axvline(x=lst_times[0], color='k', linestyle='--')
            plt.axvline(x=lst_times[-1], color='k', linestyle='--')
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y %m %d'))
            plt.show()

    def graph_X(self, year=2020, field='EWMA_ATF'):
        self.listcol = [elt for elt in self.dfx.columns if (elt[0] == field) and (elt[1].year == year)]
        self.dfxt = self.dfx[self.listcol]
        newlistcol = [elt[1] for elt in self.listcol]
        self.dfxt.columns = newlistcol
        plt.close()
        self.dfxt.plot()
        plt.show()

    def graph_XY(self, year=2019, month=12, filtertype=1, field='dt-EWMA_ATF', stage='xdt'):
        if stage == 'xdt':
            df = self.dfxtd
            listdif1 = ['EWMA_ATF', 'EWMA_FwdRatio']
            if filtertype == 1:
                for cl in [elt + '-type' for elt in listdif1]:
                    df = df.loc[df[cl] != 0]
            elif filtertype == 2:
                for cl in [elt + '-type' for elt in listdif1]:
                    df = df.loc[df[cl] != 1]
        else:
            df = self.dfxy

        # for color, either nearby:
        if field[:3] == 'dt-':
            nearby = field[3:] + '-nearby'
        elif field[:2] == 'Y-':
            nearby = field[2:] + '-nearby'
        else:
            nearby = field + '-nearby'
        # or matu
        df['matur'] = [elt[1] for elt in df.index]

        self.listcol = [elt for elt in df.columns if elt in [field, nearby]] + ['matur']

        if year == '':
            self.dfxyt = df.loc[:, self.listcol]
            newlistindex = [elt[0] for elt in df.index]
            self.dfxyt.index = newlistindex
        else:
            self.listindex = [elt for elt in df.index if (elt[1].year == year) and (elt[1].month == month)]
            self.dfxyt = df.loc[self.listindex, self.listcol]
            newlistindex = [elt[0] for elt in self.listindex]
            self.dfxyt.index = newlistindex

        self.dfxyt['time'] = self.dfxyt.index
        # self.dfxyt_pt = pd.pivot_table(self.dfxyt, values = field, index = 'time',  columns = nearby, aggfunc = np.mean)
        self.dfxyt_pt = pd.pivot_table(self.dfxyt, values=field, index='time', columns='matur', aggfunc=np.mean)

        plt.close()
        self.dfxyt_pt.plot()
        plt.show()

    def color(self, v):
        if v == 1:
            return ('black')
        elif v == -1:
            return ('red')
        else:
            return ('yellow')

    def graph_XY_scatter(self, x1, x2, y):
        self.dfxy.dropna(how='any', inplace=True)
        color = [self.color(elt) for elt in self.dfxy[y]]
        self.dfxy.plot.scatter(x=x1, y=x2, c=color, s=1)
        plt.show()
        print('average distance to center y=0 vs y=1 : ' + x1 + ', ' + x2 + ', ' + y)
        self.dfxy['d1'] = self.dfxy.apply(lambda x: abs(x[x1]), axis=1)
        self.dfxy['d2'] = self.dfxy.apply(lambda x: x[x2], axis=1)
        self.dfxy['dall'] = self.dfxy.apply(lambda x: (x[x1] ** 2 + x[x2] ** 2) ** 0.5, axis=1)
        R0 = self.dfxy.loc[self.dfxy[y] == 0][['d1', 'd2', 'dall']].mean()
        R1 = self.dfxy.loc[self.dfxy[y] == 1][['d1', 'd2', 'dall']].mean()
        print(R0)
        print(R1)

    def get_list_events(self, field='Y-EWMA_ATF', value=1):
        g.dfxy.index.names = ['Time', 'Matu']
        df = g.dfxy.copy()
        df = df.sort_index(level=['Matu', 'Time'], ascending=[True, True])
        self.rg = []
        inY = False
        for i in range(df.shape[0]):
            if not inY and (df[field][i] == value):
                inTime = i
                matu = df.index[i][1]
                inY = True
            if inY and ((df[field][i] != value) or (matu != df.index[i][1])):
                self.rg = self.rg + [df.index[inTime:i]]
                inY = False


if __name__ == '__main__':
    g = Graph('DAI')
    # g.graph_params(year=2022, month =12)
    # g.graph_inputs(year = 2020, expi_month=[4, 12], graph_month=[3], field='EWMA_FwdRatio')
    # g.graph_X(year=2019, field='TotalSignedSensiATF')
    # g.graph_XY(year=2019, month=12, field='TotalSignedSensiATF', filtertype = 1, stage='xdt')
    # g.graph_XY(year='', field='Y-EWMA_ATF', stage='xy')
    # g.graph_XY_scatter('dt-EWMA_ATF', 'dt-TotalSignedSensiATF', 'Y-EWMA_ATF')
    # g.graph_XY_scatter('dt-EWMA_FwdRatio', 'dt-TotalSensiFwdRatio', 'Y-EWMA_FwdRatio')

    g.get_list_events()
    g.graph_inputs_rg(fields=['EWMA_ATF', 'TotalSignedSensiATF'])

    # cols = ['TotalSignedSensiATF', 'TotalSensiATF', 'dt-TotalSignedSensiATF']
    # listindex = [elt for elt in g.dfxtd.index if (elt[1].year == 2019) and (elt[1].month == 12)]
    # b = g.dfxtd.loc[listindex, cols]
    # st = 2 #in days
    # lt = 20 #in days
    # st_min = int(st * 60 * 8.5)
    # lt_min = int(lt * 60 * 8.5)
    # b['TotalSignedSensiATF'] = b['TotalSignedSensiATF'].ewm(span=st_min, min_periods=st_min).mean()
    # b['TotalSensiATF'] = b['TotalSensiATF'].ewm(span=lt_min, min_periods=int(lt_min/2)).mean()
    # b[['TotalSignedSensiATF', 'TotalSensiATF']].plot(secondary_y=['TotalSignedSensiATF'])
    # plt.show()

    import pandas as pd

    folder1 = '/Users/pvamb/DBGPDS/processed'
    folder2 = '/Users/pvamb/PycharmProjects/pythonProject/processed'
    # for udl in ['SX5E', 'DAI']:
    #     for ft in ['Execs', 'UDL']:
    #         df = pd.read_pickle(folder1 + '/' + ft + '_' + udl + '.pkl')
    #         df = df.loc[df.index.to_series().dt.year==2019]
    #         df.to_pickle(folder2 + '/' + ft + '_' + udl + '.pkl')
    for udl in ['SX5E']:
        for ft in ['Execs']:
            df = pd.read_pickle(folder1 + '/' + ft + '_' + udl + '.pkl')
            df = df.loc[df.index.to_series().dt.year == 2019]
            df = df.iloc[::4, :]  # I keep onlmy in row in 4 to reduce size
            df.to_pickle(folder2 + '/' + ft + '_' + udl + '.pkl')
