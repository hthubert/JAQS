# encoding: utf-8

from __future__ import unicode_literals
import numpy as np
import pandas as pd
from pathlib import Path

from jaqs.data import DataView
from jaqs.data import RemoteDataService

from jaqs_thanf import ThanfDataService

from jaqs.research import SignalDigger
import jaqs.util as jutil
import logging
import ujson
import os

# from config_path.py, we import the path of config files
#from config_path import DATA_CONFIG_PATH
# we use read_json to read the config file to a dictionary
#data_config = jutil.read_json(DATA_CONFIG_PATH)

dataview_folder = 'output/prepared/test_signal'

UNIVERSE = Path('000300.SH.csv').read_text()
UNIVERSE = '000300.SH'
# print(UNIVERSE)


def save_dataview():
    dataview_props = {
        'start_date': 20180101,
        'end_date': 20190101,
        'universe': UNIVERSE,
        'freq': 1}
    ds = ThanfDataService()
    ds.init_from_config({"remote.data.address": "http://data.thanf.com"})
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.fields.remove('sw1')
    dv.prepare_data()

    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == '停牌'
    dv.append_df(mask_sus, 'suspended', is_quarterly=False)
    #
    dv.add_formula('not_index_member', '!index_member', is_quarterly=False)
    #
    dv.add_formula('limit_reached', 'Abs((open - Delay(close, 1)) / Delay(close, 1)) > 0.095', is_quarterly=False)

    dv.save_dataview(dataview_folder)


def analyze_signal(dv, signal_name, output_format='pdf'):
    # Step.2 calculate mask (to mask those ill data points)
    mask_sus = dv.get_ts('suspended')
    mask_index_member = dv.get_ts('not_index_member')
    mask_limit_reached = dv.get_ts('limit_reached')
    mask_all = np.logical_or(mask_sus, np.logical_or(mask_index_member, mask_limit_reached))

    signal = dv.get_ts(signal_name)  # avoid look-ahead bias
    price = dv.get_ts('close_adj')
    price_bench = dv.data_benchmark

    # Step.4 analyze!
    my_period = 5
    obj = SignalDigger(output_folder='../../output/test_signal',
                       output_format=output_format)
    obj.process_signal_before_analysis(signal, price=price,
                                       mask=mask_all,
                                       n_quantiles=5, period=my_period,
                                       #benchmark_price=price_bench,
                                       )
    res = obj.create_full_report()
    # print(res)


def simple_test_signal():
    dv = DataView()
    dv.load_dataview(dataview_folder)
    
    dv.add_formula('open_jump', 'open_adj / Delay(close_adj, 1)', is_quarterly=False) # good
    analyze_signal(dv, 'open_jump', 'pdf')
    
    print("Signal return & IC test finished.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s:[%(levelname)s]:%(message)s",
        datefmt="%Y%M%d %H:%M:%S",
        level=logging.INFO)
    save_dataview()
    simple_test_signal()
