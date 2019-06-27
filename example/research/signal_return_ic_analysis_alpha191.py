# encoding: utf-8

from __future__ import unicode_literals
import numpy as np
import pandas as pd
from pathlib import Path

from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs.research import SignalDigger
import jaqs.util as jutil

# from config_path.py, we import the path of config files
from config_path import DATA_CONFIG_PATH
# we use read_json to read the config file to a dictionary
data_config = jutil.read_json(DATA_CONFIG_PATH)

dataview_folder = 'C:/Users/ZHANGC12-L1/Desktop/JAQS/output/prepared/test_signal'
#UNIVERSE = Path('000300.SH.csv').read_text()


def save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    props = {'start_date': 20100101, 'end_date': 20190101, 'universe': '000905.SH',
             #'benchmark': '000300.SH',
             'fields': 'volume,turnover,amount',
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()

    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == '停牌'
    dv.append_df(mask_sus, 'suspended', is_quarterly=False)

    dv.add_formula('not_index_member', '!index_member', is_quarterly=False)

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
    print(price_bench)
    # Step.4 analyze!
    my_period = 5
    obj = SignalDigger(output_folder='C:/Users/ZHANGC12-L1/Desktop/JAQS/output/test_signal',
                       output_format=output_format)
    obj.process_signal_before_analysis(signal, price=price,
                                       mask=mask_all,
                                       n_quantiles=5, period=my_period,
                                       benchmark_price=price_bench,
                                       )
    res = obj.create_full_report()
    # print(res)


def simple_test_signal():
    dv = DataView()
    dv.load_dataview(dataview_folder)
    dv.add_formula('ret', '(close_adj-Delay(close_adj,1))/Delay(close_adj,1)', is_quarterly=False)
    #dv.add_formula('open_jump', 'open_adj / Delay(close_adj, 1)', is_quarterly=False) # good
    # dv.add_formula('alpha1',
    #                '-1*Corr(Rank(volume - Delay(volume, 1)),Rank((close_adj-open_adj)/open_adj),6)',
    #                is_quarterly=False)
    # dv.add_formula('alpha2',
    #                '-1*Delta(((close_adj-low_adj)-(high_adj-close_adj))/(high_adj-low_adj),1)',
    #                is_quarterly=False)
    # dv.add_formula('alpha3',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha4',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha5',
    #                '-1*Ts_Max(Corr(Ts_Rank(volume, 5), Ts_Rank(high_adj, 5), 5), 3)',
    #                is_quarterly=False)
    # dv.add_formula('alpha6',
    #                '-1*Rank(Delta(open_adj*0.85+high_adj*0.15,4))',
    #                is_quarterly=False)
    # dv.add_formula('alpha7',
    #                '(Rank(Ts_Max(vwap_adj-close_adj,3))+Rank(Ts_Min(vwap_adj-close_adj,3)))*Rank(Delta(volume,3))',
    #                is_quarterly=False)
    # dv.add_formula('alpha8',
    #                '-1*Rank(Delta((high_adj+low_adj)/2*0.2+vwap_adj*0.8,4))',
    #                is_quarterly=False)
    # dv.add_formula('alpha9',
    #                '-1*Sma(((high_adj+low_adj)/2-(Delay(high_adj,1)+Delay(low_adj,1))/2)*(high_adj-low_adj)/volume,7,2)',
    #                is_quarterly=False)
    # dv.add_formula('alpha10',
    #                ,
    #                is_quarterly=False)
    # dv.add_formula('alpha11',
    #                '-1*Ts_Sum(((close_adj-low_adj)-(high_adj-close_adj))/(high_adj-low_adj)*volume,6)',
    #                is_quarterly=False)
    # dv.add_formula('alpha12',
    #                '-1*Rank(open_adj-Ts_Sum(vwap_adj,10)/10)*Rank(Abs(close_adj-vwap_adj))',
    #                is_quarterly=False)
    # dv.add_formula('alpha13',
    #                '(high_adj*low_adj)^0.5-vwap_adj',
    #                is_quarterly=False)
    # dv.add_formula('alpha14',
    #                '-1*(close_adj-Delay(close_adj,5))',
    #                is_quarterly=False)
    # dv.add_formula('alpha15',
    #                '-1*(open_adj/Delay(open_adj,1)-1)',
    #                is_quarterly=False)
    # dv.add_formula('alpha16',
    #                '-1*Ts_Max(Rank(Corr(Rank(volume),Rank(vwap_adj),5)),5)',
    #                is_quarterly=False)
    # dv.add_formula('alpha17',
    #                '-1*Rank(vwap_adj-Ts_Max(vwap_adj,15))^Delta(close_adj,5)',
    #                is_quarterly=False)
    # dv.add_formula('alpha18',
    #                '-1*close_adj/Delay(close_adj,5)',
    #                is_quarterly=False)
    # dv.add_formula('alpha19',
    #                ,
    #                is_quarterly=False)
    # dv.add_formula('alpha20',
    #                '-1*(close_adj-Delay(close_adj,6))/Delay(close_adj,6)*100',
    #                is_quarterly=False)
    # dv.add_formula('alpha21',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha22',
    #                '-1*Sma((close_adj-Ts_Mean(close_adj,6))/Ts_Mean(close_adj,6)-Delay((close_adj-Ts_Mean(close_adj,6))/Ts_Mean(close_adj,6),3),12,1)',
    #                is_quarterly=False)
    # dv.add_formula('alpha23',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha24',
    #                '-1*Sma(close_adj-Delay(close_adj, 5), 5, 1)',
    #                is_quarterly=False)
    # dv.add_formula('alpha25',
    #                'Rank(Delta(close_adj,7)*(1-Rank(Decay_linear(volume/Ts_Mean(volume,20),9))))*(1+Rank(Ts_Sum(ret,250)))',
    #                is_quarterly=False)
    # dv.add_formula('alpha26',
    #                'Ts_Sum(close_adj,7)/7-close_adj+Corr(vwap_adj,Delay(close_adj,5),230)',
    #                is_quarterly=False)
    # dv.add_formula('alpha27',
    #                '-1*Ewma((close_adj-Delay(close_adj,3))/Delay(close_adj,3)*100+(close_adj-Delay(close_adj,6))/Delay(close_adj,6)*100,12)',
    #                is_quarterly=False)
    # dv.add_formula('alpha28',
    #                '-1*(3*Sma((close_adj-Ts_Min(low_adj,9))/(Ts_Max(high_adj,9)-Ts_Min(low_adj,9))*100,3,1)-2*Sma(Sma((close_adj-Ts_Min(low_adj,9))/(Ts_Max(high_adj,9)-Ts_Min(low_adj,9))*100,3,1),3,1))',
    #                is_quarterly=False)
    # dv.add_formula('alpha29',
    #                '-1*(close_adj-Delay(close_adj,6))/Delay(close_adj,6)*volume',
    #                is_quarterly=False)
    # dv.add_formula('alpha30',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha31',
    #                '-1*(close_adj-Ts_Mean(close_adj,12))/Ts_Mean(close_adj,12)*100',
    #                is_quarterly=False)
    # dv.add_formula('alpha32',
    #                '-1*Ts_Sum(Rank(Corr(Rank(high_adj),Rank(volume),3)),3)',
    #                is_quarterly=False)
    # dv.add_formula('alpha33',
    #                '(Delay(Ts_Min(low_adj,5),5)-Ts_Min(low_adj,5))*Rank((Ts_Sum(ret,240)-Ts_Sum(ret,20))/220)*Ts_Rank(volume,5)',
    #                is_quarterly=False)
    # dv.add_formula('alpha34',
    #                'Ts_Mean(close_adj,12)/close_adj',
    #                is_quarterly=False)
    # dv.add_formula('alpha35',
    #                '-1*Min(Rank(Decay_linear(Delta(open_adj,1),15)),Rank(Decay_linear(Corr(volume,close_adj*0.65+open_adj*0.35,17),7)))',
    #                is_quarterly=False)
    # dv.add_formula('alpha36',
    #                '-1*Rank(Ts_Sum(Corr(Rank(volume),Rank(vwap_adj),6),2))',
    #                is_quarterly=False)
    # dv.add_formula('alpha37',
    #                '-1*Rank(Ts_Sum(open_adj,5)*Ts_Sum(ret,5)-Delay(Ts_Sum(open_adj,5)*Ts_Sum(ret,5),10))',
    #                is_quarterly=False)
    # dv.add_formula('alpha38',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha39',
    #                '-1*(Rank(Decay_linear(Delta(close_adj,2),8))-Rank(Decay_linear(Corr((vwap_adj*0.3+open_adj*0.7),Ts_Sum(Ts_Mean(volume,180),37),14),12)))',
    #                is_quarterly=False)
    # dv.add_formula('alpha40',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha41',
    #                '-1*Rank(Ts_Max(Delta(vwap_adj,3),5))',
    #                is_quarterly=False)
    # dv.add_formula('alpha42',
    #                '-1*Rank(StdDev(high_adj,10))*Corr(high_adj,volume,10)',
    #                is_quarterly=False)
    # dv.add_formula('alpha43',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha44',
    #                '-1*(Ts_Rank(Decay_linear(Corr(low_adj,Ts_Mean(volume,10),7),6),4)+Ts_Rank(Decay_linear(Delta(vwap_adj,3),10),15))',
    #                is_quarterly=False)
    # dv.add_formula('alpha45',
    #                '-1*Rank(Delta(close_adj*0.6+open_adj*0.4,1))*Rank(Corr(vwap_adj,Ts_Mean(volume,150),15))',
    #                is_quarterly=False)
    # dv.add_formula('alpha46',
    #                '(Ts_Mean(close_adj,3)+Ts_Mean(close_adj,6)+Ts_Mean(close_adj,12)+Ts_Mean(close_adj,24))/4/close_adj',
    #                is_quarterly=False)
    # dv.add_formula('alpha47',
    #                'Sma((Ts_Max(high_adj,6)-close_adj)/(Ts_Max(high_adj,6)-Ts_Min(low_adj,6))*100,9,1)',
    #                is_quarterly=False)
    # dv.add_formula('alpha48',
    #                '-1*Rank(Sign(close_adj-Delay(close_adj,1))+Sign(Delay(close_adj,1)-Delay(close_adj,2))+Sign(Delay(close_adj,2)-Delay(close_adj,3)))*Ts_Sum(volume,5)/Ts_Sum(volume,20)',
    #                is_quarterly=False)
    # dv.add_formula('alpha49',
    #                ',
    #                is_quarterly=False)
    # dv.add_formula('alpha50',
    #                ',
    #                is_quarterly=False)
    analyze_signal(dv, 'alpha48', 'pdf')
    
    print("Signal return & IC test finished.")


if __name__ == "__main__":
    save_dataview()
    simple_test_signal()
