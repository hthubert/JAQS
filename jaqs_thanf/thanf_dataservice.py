# encoding: UTF-8
"""
Module thanf_dataservice defines ThanfDataService.

DataService is just an interface. ThanfDataService is a wrapper class for DataApi.
It inherits all methods of DataApi and implements several convenient methods making
query data more natural and easy.

"""

from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from jaqs.data.dataservice import *
from .dataclient import ThanfDataClient


class ThanfDataService(with_metaclass(Singleton, DataService)):
    """

    """

    def __init__(self):
        # print("Init ThanfDataService DEBUG")
        super(ThanfDataService, self).__init__()
        self._client = None
        self._address = ""
        self._username = ""
        self._password = ""
        self._trade_dates_df = None
        self._inst_columns = [
            'listed_date',
            'industry_name',
            'round_lot',
            'de_listed_date',
            'tick_size',
            'symbol',
            'market_tplus',

        ]
        self._REPORT_DATE_FIELD_NAME = 'report_date'

    def init_from_config(self, props, start_date='20100101', end_date=''):
        """

        Parameters
        ----------
        :param start_date:
        :param end_date:
        :param props: dict
            Configurations used for initialization.

        Example
        -------
        {"remote.data.address": "tcp://Address:Port",
        "remote.data.username": "your username",
        "remote.data.password": "your password"}


        """

        def get_from_list_of_dict(l, key, default=None):
            res = None
            for dic in l:
                res = dic.get(key, None)
                if res is not None:
                    break
            if res is None:
                res = default
            return res

        props_default = dict()
        dic_list = [props, props_default]

        address = get_from_list_of_dict(dic_list, "remote.data.address", "")
        username = get_from_list_of_dict(dic_list, "remote.data.username", "")
        password = get_from_list_of_dict(dic_list, "remote.data.password", "")

        self._address = address
        self._username = username
        self._password = password
        self._client = ThanfDataClient(self._address)

        trade_dates_df, err_msg = self._client.query_trade_dates(start_date, end_date)
        if not trade_dates_df.empty:
            self._trade_dates_df = trade_dates_df
        else:
            print("No trade date.\n".format(err_msg))

        return err_msg

    @staticmethod
    def _raise_error_if_msg(err_msg):
        if err_msg is None:
            return
        items = err_msg.split(',')
        if not (items and (items[0] == '0')):
            raise QueryDataError(err_msg)

    # -----------------------------------------------------------------------------------
    # Basic APIs
    def quote(self, symbol, fields=""):
        pass

    def bar_quote(self,
                  symbol,
                  start_time=200000,
                  end_time=160000,
                  trade_date=0,
                  freq="1M",
                  fields="",
                  data_format="",
                  **kwargs):
        pass

    def daily(self, symbol, start_date, end_date,
              fields="", adjust_mode=None):
        """
        Query dar bar,
        support auto-fill suspended securities data,
        support auto-adjust for splits, dividends and distributions.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_date : int or str
            YYYMMDD or 'YYYY-MM-DD'
        end_date : int or str
            YYYMMDD or 'YYYY-MM-DD'
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        adjust_mode : str or None, optional
            None for no adjust;
            'pre' for forward adjust;
            'post' for backward adjust.

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, trade_date, open, high, low, close, volume, turnover, vwap, oi, suspended
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.daily("00001.SH,cu1709",start_date=20170503, end_date=20170708,
                            fields="open,high,low,last,volume", fq=None, skip_suspended=True)

        """

        df, err_msg = self._client.daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust_mode=adjust_mode)
        self._raise_error_if_msg(err_msg)
        columns = df.columns.values.tolist()
        for x in fields.split(','):
            if (not x.endswith('_adj')) and (x not in columns):
                df[x] = 0.0

        return df, err_msg

    def bar(self, symbol, start_time=200000, end_time=160000, trade_date=None, freq='1M', fields=""):
        pass

    def query(self, view, filter, fields):
        pass

    # ---------------------------------------------------------------------
    # Calendar
    def query_trade_dates(self, start_date, end_date):
        """
        Get array of trade dates within given range.
        Return zero size array if no trade dates within range.

        Parameters
        ----------
        start_date : int
            YYmmdd
        end_date : int

        Returns
        -------
        trade_dates_arr : np.ndarray
            dtype = int

        """
        df_raw = self._trade_dates_df[self._trade_dates_df['trade_date'] >= str(start_date)]
        df_raw = df_raw[df_raw['trade_date'] <= str(end_date)]

        if df_raw.empty:
            return np.array([], dtype=int)

        trade_dates_arr = df_raw['trade_date'].values.astype(np.integer)
        return trade_dates_arr

    def query_last_trade_date(self, date):
        """

        Parameters
        ----------
        date : int

        Returns
        -------
        res : int

        """
        dt = jutil.convert_int_to_datetime(date)
        delta = pd.Timedelta(weeks=2)
        dt_old = dt - delta
        date_old = jutil.convert_datetime_to_int(dt_old)

        dates = self.query_trade_dates(date_old, date)
        mask = dates < date
        res = dates[mask][-1]

        return int(res)

    def is_trade_date(self, date):
        """
        Check whether date is a trade date.

        Parameters
        ----------
        date : int

        Returns
        -------
        bool

        """
        dates = self.query_trade_dates(date, date)
        return len(dates) > 0

    def query_next_trade_date(self, date, n=1):
        """

        Parameters
        ----------
        date : int
        n : int, optional
            Next n trade dates, default 0 (next trade date).

        Returns
        -------
        res : int

        """
        dt = jutil.convert_int_to_datetime(date)
        delta = pd.Timedelta(weeks=(n // 7 + 2))
        dt_new = dt + delta
        date_new = jutil.convert_datetime_to_int(dt_new)

        dates = self.query_trade_dates(date, date_new)
        mask = dates > date
        res = dates[mask][n - 1]

        return int(res)

    # -----------------------------------------------------------------------------------
    # Convenient Functions
    @staticmethod
    def _dict2url(d):
        """
        Convert a dict to str like 'k1=v1&k2=v2'

        Parameters
        ----------
        d : dict

        Returns
        -------
        str

        """
        items = ['='.join([key, str(value)]) for key, value in d.items()]
        return '&'.join(items)

    def query_universe_member(self, universe, start_date, end_date):
        """
        Return list of symbols that have been in index during start_date and end_date.

        Parameters
        ----------
        universe : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        list

        """
        return [universe]

    def query_index_member(self, index, start_date, end_date):
        """
        Return list of symbols that have been in index during start_date and end_date.

        Parameters
        ----------
        index : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        list

        """
        return [index]

    def query_lb_dailyindicator(self, symbol, start_date, end_date, fields=""):
        """
        Helper function to call data_api.query with 'lb.secDailyIndicator' more conveniently.

        Parameters
        ----------
        symbol : str
            separated by ','
        start_date : int
        end_date : int
        fields : str, optional
            separated by ',', default ""

        Returns
        -------
        df : pd.DataFrame
            index date, columns fields
        err_msg : str

        """
        pass

    def query_inst_info(self, symbol, inst_type="", fields=""):
        df_raw, err_msg = self._client.query_inst_info(symbol, self._inst_columns)
        self._raise_error_if_msg(err_msg)
        dtype_map = {'symbol': str, 'list_date': np.integer, 'delist_date': np.integer, 'inst_type': np.integer}
        cols = set(df_raw.columns)
        dtype_map = {k: v for k, v in dtype_map.items() if k in cols}

        df_raw = df_raw.astype(dtype=dtype_map)

        res = df_raw.set_index('symbol')
        return res
        pass
