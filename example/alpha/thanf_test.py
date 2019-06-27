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

from jaqs.trade import model
from jaqs.trade import (AlphaStrategy, AlphaBacktestInstance, AlphaTradeApi,
                        PortfolioManager, AlphaLiveTradeInstance, RealTimeTradeApi)

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
# UNIVERSE = '600559.SH'


def save_data():
    """
    This function fetches data from remote server and stores them locally.
    Then we can use local data to do back-test.

    """
    dataview_props = {'start_date': 20170101,  # Start and end date of back-test
                      'end_date': 20171030,
                      'symbol': UNIVERSE,  # Investment universe and performance benchmark
                      'benchmark': '000300.SH',
                      # 'fields': 'total_mv,turnover',  # Data fields that we need
                      'freq': 1  # freq = 1 means we use daily data. Please do not change this.
                      }
    # ThanfDataService communicates with a remote server to fetch data
    ds = ThanfDataService()

    # Use username and password in data_config to login
    ds.init_from_config({"remote.data.address": "http://data.thanf.com"})

    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.fields.remove('sw1')
    dv.prepare_data()
    dv.save_dataview(folder_path=dataview_store_folder)


def do_backtest():
    # Load local data file that we just stored.
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder, large_memory=False)

    backtest_props = {"start_date": dv.start_date,  # start and end date of back-test
                      "end_date": dv.end_date,
                      "period": "month",  # re-balance period length
                      "benchmark": dv.benchmark,  # benchmark and universe
                      "universe": dv.universe,
                      "init_balance": 1e8,  # Amount of money at the start of back-test
                      "position_ratio": 1.0,  # Amount of money at the start of back-test
                      }
    backtest_props.update(data_config)
    backtest_props.update(trade_config)

    # Create model context using AlphaTradeApi, AlphaStrategy, PortfolioManager and AlphaBacktestInstance.
    # We can store anything, e.g., public variables in context.

    trade_api = AlphaTradeApi()
    strategy = AlphaStrategy()
    pm = PortfolioManager()
    bt = AlphaBacktestInstance()
    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm)

    bt.init_from_config(backtest_props)
    bt.run_alpha()

    # After finishing back-test, we save trade results into a folder
    bt.save_results(folder_path=backtest_result_folder)


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
