# encoding: UTF-8
"""
Module dataclient defines ThanfDataClient.
"""

from __future__ import print_function
from __future__ import unicode_literals

import os
import asyncio
import datetime
import ujson
import pandas as pd
import requests
import logging
from . import utils


class ThanfDataClient(object):

    def __init__(self, address="http://data.thanf.com"):
        self._address = address
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
        self._index_weights_columns = [
            'trade_date',
            'symbol',
            'sec_name',
            'weight',
            'index_code']
        self.adj_factor_columns = [
            'symbol',
            'trade_date',
            'adjust_factor']
        self._index_member_columns = [
            'in_date',
            'out_date',
            'symbol'
        ]

    @staticmethod
    def _parse_error(content):
        err = eval(content)
        return "{0},{1}".format(err.get('error_code'), err.get('error_msg'))

    @staticmethod
    def _parse(content):
        try:
            if content.startswith(b"{'error_code'"):
                return None, ThanfDataClient._parse_error(content)
            else:
                return ujson.loads(content), None
        except ValueError:
            with open(os.path.join(os.getcwd(), "error.txt"), 'wb') as f:
                f.write(content)
            raise

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
    def _get_response_async(urls: list):
        async def get_json(url):
            json = await loop.run_in_executor(None, requests.get, url)
            responses.append(json)

        async def run(items: list):
            await asyncio.gather(*[get_json(x) for x in items])

        loop = asyncio.new_event_loop()
        responses = []
        for i in utils.chunks(urls, max(len(urls) // 1, 1)):
            loop.run_until_complete(run(i))
        return responses

    @staticmethod
    def _get_response(urls: list):
        logging.info('get_response begin')
        responses = []
        with requests.Session() as s:
            for i in urls:
                responses.append(s.get(i))
        logging.info('get_response end')
        return responses

    def _parse_bar(self, data: list):
        if len(data) == 0 or len(data[0]) == len(self._bar_columns):
            df = pd.DataFrame(data, columns=self._bar_columns)
            df['trade_status'] = df['trade_status'].apply(lambda x: '停牌' if x != '交易' else x)
        else:
            df = pd.DataFrame(data, columns=self._bar_columns[:-1])
            df['trade_status'] = '交易'
        df['vwap'] = 0
        df.loc[df['volume'] > 0, 'vwap'] = df['turnover']/df['volume']
        return df

    def _parse_daily_rsp(self, rsp_list: list):
        logging.info('parse_daily_rsp begin')
        rsp_list = [self._parse(x.content) for x in rsp_list]
        df = pd.DataFrame(columns=self._bar_columns)
        err_msg = None
        for data, err in rsp_list:
            if data is not None:
                df = df.append(self._parse_bar(data), sort=False)
            else:
                err_msg = err
        logging.info('parse_daily_rsp end')
        return df, err_msg

    def daily(self, symbol: str, start_date, end_date, adjust_mode=None):
        urls = list(map(
            lambda x: self._get_bar_url(x, start_date, end_date, "D", adjust_mode),
            symbol.split(',')))

        return self._parse_daily_rsp(self._get_response(urls))

    def query_inst_info(self, symbol, fields: list):
        url = '{0}/instruments?order_book_id={1}'.format(self._address, symbol)
        data, err_msg = self._parse(requests.get(url).content)
        return pd.DataFrame(data)[fields], err_msg

    def query_index_weights_range(self, index, start_date, end_date):
        args = utils.dict2url({
            'order_book_id': index,
            'start_date': start_date,
            'end_date': end_date})
        url = '{0}/get_index_component_info?{1}'.format(self._address, args)
        data, err_msg = self._parse(requests.get(url).content)
        return pd.DataFrame(data, columns=self._index_weights_columns), err_msg

    def _get_adj_factor_url(self, symbol, start_date, end_date):
        url_pattern = "{0}/get_stock_adjfactor?{1}"
        items = {
            "order_book_id": symbol,
            "start_date": start_date,
            "end_date": end_date
        }
        return url_pattern.format(self._address, utils.dict2url(items))

    def _parse_adj_factor(self, data: list):
        return pd.DataFrame(data, columns=self.adj_factor_columns)

    def _parse_adj_factor_rsp(self, rsp_list: list):
        rsp_list = [self._parse(x.content) for x in rsp_list]
        df = pd.DataFrame(columns=self.adj_factor_columns)
        err_msg = None
        for data, err in rsp_list:
            if data is not None:
                df = df.append(self._parse_adj_factor(data))
            else:
                err_msg = err

        return df, err_msg

    def query_adj_factor(self, symbol, start_date, end_date):
        urls = list(map(
            lambda x: self._get_adj_factor_url(x, start_date, end_date),
            symbol.split(',')))
        return self._parse_adj_factor_rsp(self._get_response(urls))

    def query_index_member(self, index, start_date, end_date):
        url_pattern = "{0}/get_index_component_transfer_info?{1}"
        args = utils.dict2url({
            'order_book_id': index,
            'start_date': start_date,
            'end_date': end_date})

        rsp = requests.get(url_pattern.format(self._address, args))
        data, err_msg = self._parse(rsp.content)
        return pd.DataFrame(data, columns=self._index_member_columns), err_msg
