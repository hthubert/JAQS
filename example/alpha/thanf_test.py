# encoding: utf-8

"""
A very first example of AlphaStrategy back-test:
    Market value weight among UNIVERSE.
    Benchmark is HS300.

"""

from __future__ import print_function, unicode_literals, division, absolute_import
from pathlib import Path
from jaqs.data import DataView
from jaqs_thanf import ThanfDataService

import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH

data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

# Data files are stored in this folder:
dataview_store_folder = '../../output/simplest/dataview'

# Back-test and analysis results are stored here
backtest_result_folder = '../../output/simplest'

# UNIVERSE = '000807.SH'
UNIVERSE = Path('000807.txt').read_text()


def save_data():
    """
    This function fetches data from remote server and stores them locally.
    Then we can use local data to do back-test.

    """
    dataview_props = {'start_date': 20170101,  # Start and end date of back-test
                      'end_date': 20171030,
                      'symbol': UNIVERSE,  # Investment universe and performance benchmark
                      'benchmark': '000300.SH',
                      # 'fields': 'total_mv,turnover,tot_oper_cost',  # Data fields that we need
                      'freq': 1  # freq = 1 means we use daily data. Please do not change this.
                      }
    # ThanfDataService communicates with a remote server to fetch data
    ds = ThanfDataService()

    # Use username and password in data_config to login
    ds.init_from_config({"remote.data.address": "http://data.thanf.com"}, '20170101', '20171231')

    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.prepare_data()
    pass


def do_backtest():
    pass


def analyze_backtest_results():
    pass


def do_livetrade():
    pass


if __name__ == "__main__":
    is_backtest = True

    if is_backtest:
        save_data()
        do_backtest()
        analyze_backtest_results()
    else:
        save_data()
        do_livetrade()
