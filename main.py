from SetUp import *

# from GetRawData import get_raw_data
# get_raw_data()
#
# from PricingAndCalibration import Pricing, Fitting
#
# P = Pricing()
#
# for udl in stocks_list:
#     print(udl)
#     res = pd.DataFrame()
#
#     for MaturityDate in dates_expi:  # [pd.Timestamp('2022-12-16 15:30:00')]: #dates_expi:  #[pd.Timestamp('2019-04-19 15:30:00')]:
#         print(MaturityDate)
#         fit = Fitting(folder1, udl, MaturityDate)
#         if fit.bigEnough:
#             fit.clusterize()
#
#             while (fit.cluster.shape[0] > 0) and (
#                     fit.df.loc[fit.last_index, 'timeOfTrade'] < MaturityDate - datetime.timedelta(5)):
#                 fit.reref()
#                 fit.price_cluster(udl)
#                 possible = fit.get_new_fwd_ratio()
#                 if possible:
#                     oldATF = fit.ATF
#                     oldSMI = fit.SMI
#                     oldCVX = fit.CVX
#                     fit.get_new_vols_params()
#                     if (fit.min_nb_opt_per_cluster == fit.start_min_nb_opt_per_cluster) and \
#                             ((abs(fit.ATF - oldATF) > fit.stdParams[0] * 4) or (
#                                     abs(fit.SMI - oldSMI) > fit.stdParams[1] * 6) or (
#                                      abs(fit.CVX - oldCVX) > fit.stdParams[2] * 8)):
#                         print('lets recompute the fwd ratio')
#                         print(fit.ATF, oldATF, fit.SMI, oldSMI, fit.CVX, oldCVX)
#                         fit.min_nb_opt_per_cluster += 1
#                     else:
#                         fit.write_down()
#                         fit.min_nb_opt_per_cluster = fit.start_min_nb_opt_per_cluster
#                 else:
#                     fit.min_nb_opt_per_cluster += 1
#
#                 # print(fit.df.loc[fit.start_index, 'timeOfTrade'])
#                 # if fit.df.loc[fit.start_index, 'timeOfTrade'] > pd.Timestamp('2018-02-07 00:00:00'):
#                 #     j=7
#                 #     print('her')
#
#                 fit.clusterize()
#
#             fit.compute_EWMA()  # halftime in hours
#             if res.shape[0] == 0:
#                 res = fit.df_params.copy()
#             else:
#                 res = res.append(fit.df_params, ignore_index=True)
#             print(res.tail(10))
#
#     # Filter out if Error is too big
#     before = res.shape[0]
#     compare_range = max(30, int(before / 100 / 2))
#     res.to_pickle(folder2 + '/Parameters_before_filter_' + udl + '.pkl')
#     res = res.sort_values(by='StartTime', ascending=True)
#     global_mean = res.Error.mean()
#     res['maxE'] = res.Error.rolling(compare_range * 2).mean().shift(periods=-compare_range, fill_value=0).apply(
#         lambda x: max(global_mean, x)) * (2 + 3 * res.TTM.apply(lambda x: min(2, x)))
#     res = res.loc[res.Error < res.maxE]
#     print('Pct rows out ' + str(1 - res.shape[0] / before))
#     # take out TTM column
#     del res['TTM']
#
#     res.to_pickle(folder2 + '/Parameters_' + udl + '.pkl')
#
# from BuildInputs import BuildInputs
#
# print('BuildInputs')
# for udl in stocks_list:
#     print(udl)
#     df = pd.DataFrame()
#     for pos, matu in enumerate(dates_expi):
#         # if matu == pd.Timestamp('2020-06-19 00:00:00'):
#         print(matu)
#         build = BuildInputs(udl, matu)
#         if build.df_params.shape[0] > 10:
#             build.even_index()
#             build.get_total_sensi()
#             build.merge()
#             df = df.append(build.df)
#
#     df.to_pickle(folder2 + '/Inputs_' + udl + '.pkl')
#
# from Graph import Graph
# g = Graph('SX5E')
# g.graph_params(2020,9)
# g.graph_inputs(2018=9,'EWMA_CVX')
#
#
from BuildXY import Data
#
# print('BuildXY')
# for udl in stocks_list:
#     print(udl)
#     data = Data(udl)
#     data.differentiate_matu()
#     data.df_pivot.to_pickle(folder3 + '/X_' + udl + '.pkl')

for udl in stocks_list:
    print(udl)
    data = Data(udl)
    data.df_pivot = pd.read_pickle(folder3 + '/X_' + udl + '.pkl')
    data.differentiate_time(st, lt, Ylag)  # ...and compute Y
    data.X.to_pickle(folder3 + '/Xtd_' + udl + '.pkl')

for udl in stocks_list:
    data = Data(udl)

    data.X = pd.read_pickle(folder3 + '/Xtd_' + udl + '.pkl')

    print('filter')
    data.filter(TTM=1, type=filter_type)  # TTM in years;  type in 1:fwd only, 2:nearby only, 3: both

    if udl not in [ref]:
        XRef = pd.read_pickle(folder3 + '/XY_' + ref + '.pkl')

        data.differentiate_refindex(XRef, exclude=['FwdRatio', 'TotalSignedSensi'])

        data.normalize(cap)
        # is it better to normalize before or after joining the underlyings? Both have pros and cons.

        data.X.to_pickle(folder3 + '/XY_' + udl + '.pkl')
    else:
        data.X.to_pickle(folder3 + '/XY_' + udl + '.pkl')

df = pd.DataFrame()
for udl in [elt for elt in stocks_list if elt not in indexlist]:
    dft = pd.read_pickle(folder3 + '/XY_' + udl + '.pkl')
    dft['udl'] = udl
    df = df.append(dft)
df.to_pickle(folder3 + '/XY_all_stocks -st_' + str(st) + '-lt_' + str(lt) + '-type_' + str(filter_type) + '-cap_' + str(
    cap) + '.pkl')
