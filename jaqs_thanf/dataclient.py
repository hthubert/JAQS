# encoding: UTF-8
"""
Module dataclient defines ThanfDataClient.
"""

from __future__ import print_function
from __future__ import unicode_literals

import asyncio
import requests
import datetime
import ujson
import pandas as pd
from . import utils


class ThanfDataClient(object):

    def __init__(self, addr="http://data.thanf.com"):
        self._address = addr
        self._bar_columns = [
            'symbol',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'turnover',
            'trade_status']

    @staticmethod
    def _parse_error(content):
        err = eval(content)
        return "{0},{1}".format(err.get('error_code'), err.get('error_msg'))

    @staticmethod
    def _parse(content):
        if content.startswith(b"{'error_code'"):
            return None, ThanfDataClient._parse_error(content)
        else:
            return ujson.loads(content), None

    def query_trade_dates(self, start_date, end_date):
        if start_date == "":
            start_date = "20100101"
        if end_date == "":
            end_date = "{0}1231".format(datetime.datetime.today().year)
        params = {'start_date': start_date, "end_date": end_date}
        r = requests.get("{0}/trading_dates".format(self._address), params)
        dates, err_msg = self._parse(r.content)
        columnset = {"istradeday": ['T'] * len(dates), "trade_date": dates}
        return utils.to_dataframe(columnset), err_msg

    def _get_bar_url(self, symbol, start_date, end_date, ktype, atype):
        url_pattern = "{0}/get_history_bars?{1}"
        items = {
            "order_book_id": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "ktype": ktype,
            "atype": utils.get_atype(atype)
        }
        return url_pattern.format(self._address, utils.dict2url(items))

    @staticmethod
    def _get_response(urls: list):
        async def run(items: list):
            futures = [loop.run_in_executor(None, requests.get, x) for x in items]
            responses.extend([await f for f in futures])

        loop = asyncio.get_event_loop()
        responses = []

        for i in utils.chunks(urls, max(len(urls) // 4, 1)):
            loop.run_until_complete(run(i))

        return responses

    def _parse_bar(self, data: list):
        return pd.DataFrame(data, columns=self._bar_columns)

    def _parse_daily_rsp(self, rsp_list: list):
        rsp_list = [self._parse(x.content) for x in rsp_list]
        df = pd.DataFrame(columns=self._bar_columns)
        err_msg = None
        for data, err in rsp_list:
            if data is not None:
                df = df.append(self._parse_bar(data))
            else:
                err_msg = err

        return df, err_msg

    def daily(self, symbol: str, start_date, end_date, adjust_mode=None):
        urls = list(map(
            lambda x: self._get_bar_url(x, start_date, end_date, "D", adjust_mode),
            symbol.split(',')))

        return self._parse_daily_rsp(self._get_response(urls))

    def query_inst_info(self, symbol, fields: list):
        url = "{0}/instruments?order_book_id={1}".format(self._address, symbol)
        data, err_msg = self._parse(requests.get(url).content)
        return pd.DataFrame(data)[fields], err_msg
