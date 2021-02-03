import numpy as np
import pandas as pd
import os
import datetime
import glob

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
import math
import scipy.stats as si
import statsmodels.api as sm
import matplotlib.pyplot as plt

# local_data_folder = '/Users/pvamb/DBGPDS/deutsche-boerse-xetra-pds'  # do not end in /
# local_data_folder_opt = '/Users/pvamb/DBGPDS/deutsche-boerse-eurex-pds'  # do not end in /
# folder1 = '/Users/pvamb/DBGPDS/processed'
# folder2 = '/Users/pvamb/DBGPDS/parameters'
# folder3 = '/Users/pvamb/DBGPDS/XY'
# folder4 = '/Users/pvamb/DBGPDS/MLoutput'

folder1 = 'D:/Users/GitHub/DBG-PDS/processed'
folder2 = 'D:/Users/GitHub/DBG-PDS/parameters'
folder3 = 'D:/Users/GitHub/DBG-PDS/XY'
folder4 = 'D:/Users/GitHub/DBG-PDS/MLoutput'

# folder1 = '/root/docker/DBGPDS/processed'
# folder2 = '/root/docker/DBGPDS/parameters'
# folder3 = '/root/docker/DBGPDS/XY'
# folder4 = '/root/docker/DBGPDS/MLoutput'


opening_hours_str = "07:00"
closing_hours_str = "15:30"
time_fmt = "%H:%M"
opening_hours = datetime.datetime.strptime(opening_hours_str, time_fmt).time()
closing_hours = datetime.datetime.strptime(closing_hours_str, time_fmt).time()

stocks_list0 = ['DBK', 'EOAN', 'CBK', 'DTE', 'SVAB', 'RWE', 'IFX', 'LHA', 'DAI', 'TKA',
                'HDD', 'O2D', 'EVT', 'AIXA', 'DPW', 'SIE', 'PSM', 'BAS', 'BAYN', 'SAP', 'BMW', 'SDF',
                'VOW3', 'FRE', 'AB1', 'CEC', 'GAZ', 'VNA', 'SHA', 'B4B', 'UN01', 'ALV', 'NDX1',
                'DLG', 'ADV', 'AT1', 'NOA3', 'VODI', 'BPE5', 'HEI', 'ADS', 'KCO', 'TUI1', 'SZU',
                'DEZ', 'EVK', 'WDI', 'MRK', 'PAH3', 'G1A', 'MUV2', 'QSC', 'HEN3', 'QIA', 'TINA',
                'DWNI', 'ANO', 'ZAL', 'RKET', 'SGL', 'FME', 'IGY', '1COV', 'BVB', 'FNTN', 'DB1',
                'PBB', 'LIN', 'CON', 'UTDI', 'KGX', 'EV4', 'TEG', 'PNE3', 'OSR', 'BEI', 'LLD', 'ARL',
                'MDG1' 'LXS' 'BNR' 'GYC' 'ZIL2' 'SANT' 'AOX' 'DRI' 'TTI' 'BOSS' 'SZG'
                'RIB', 'ABR', 'DEQ', 'SOW', 'CAP', 'WAF', 'SY1', 'GBF', 'NDA', 'ADE']
stocks_list1 = ['DAI', 'CBK', 'DBK', 'DTE', 'EOAN']  # 'DAI'
stocks_list2 = ['RWE', 'IFX', 'LHA', 'TKA', 'DPW', 'SIE', 'BAS', 'BAYN', 'SAP', 'BMW']
stocks_list3 = ['VOW3', 'FRE', 'VNA', 'ALV']

ref = 'SX5E'
indexlist = ['SX5E']

stocks_list = ['DAI']  # + stocks_list2 + stocks_list3

bank_h = ['01-01-2020', '10-04-2020', '13-04-2020', '01-05-2020', '01-06-2020', '24-12-2020', '25-12-2020',
          '31-12-2020',
          '01-01-2019', '19-04-2019', '22-04-2019', '01-05-2019', '24-12-2019', '25-12-2019', '26-12-2019',
          '31-12-2019',
          '01-01-2018', '02-04-2018', '01-05-2018', '21-05-2018', '03-10-2018', '24-12-2018', '25-12-2018',
          '26-12-2018',
          '31-12-2018', '14-04-2017', '17-04-2017', '01-05-2017', '05-06-2017', '03-10-2017', '31-10-2017',
          '25-12-2017',
          '26-12-2017']
bank_h_ts = np.array([np.datetime64(pd.Timestamp(elt).date()) for elt in bank_h])


def time_between(a, b):
    nbd = np.busday_count(a.date(), b.date(), holidays=bank_h_ts)
    TimeA = datetime.datetime.combine(datetime.date.today(), a.time())
    TimeB = datetime.datetime.combine(datetime.date.today(), b.time())
    if TimeB > TimeA:
        addhours = (TimeB - TimeA).total_seconds() / 3600
    else:
        addhours = -((TimeA - TimeB).total_seconds() / 3600)
    return (nbd + addhours / 8.5) / 252


def get_last_working(dt):
    while dt in bank_h_ts:
        dt = dt - datetime.timedelta(1)
    return (dt)


from_date = '2017-01-01'
until_date = '2020-10-02'
last_matu = '2022-12-31'
dates = list(pd.date_range(from_date, until_date, freq='D').strftime('%Y-%m-%d'))
dates_expi = list(pd.date_range(from_date, last_matu, freq='W'))
dates_expi = [elt - datetime.timedelta(2) for elt in dates_expi]
dates_expi = [datetime.datetime.combine(elt, closing_hours) for elt in dates_expi if
              elt.day in [15, 16, 17, 18, 19, 20, 21]]
dates_expi = [get_last_working(elt) for elt in dates_expi]
dates_expi_trim = [elt for elt in dates_expi if elt.month in [3, 6, 9, 12]]

st = 2  # in days
lt = 20  # in days
Ylag = 5  # in days

filter_type = 1
cap = 3
