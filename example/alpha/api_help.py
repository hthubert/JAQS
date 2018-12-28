# encoding: utf-8

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
import jaqs.util as jutil

data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

from jaqs_fxdayu.data import DataApi

if __name__ == "__main__":
    api = DataApi(data_config["remote.data.address"])  # 传入连接到的远端数据服务器的tcp地址
    api.login(username=data_config["remote.data.username"],
              password=data_config["remote.data.password"])

    df, msg = api.query(
        view="help.apiList",
        fields="",
        filter="")
    df, msg = api.query(view="help.apiParam", fields="", filter="api=jz.instrumentInfo")
