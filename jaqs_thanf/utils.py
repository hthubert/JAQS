from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import datetime as dt
import pandas as pd
import numpy as np
import math
import re

long_nan = 9223372036854775807


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def dict2url(d):
    """
    Convert a dict to str like 'k1=v1&k2=v2'

    Parameters
    ----------
    d : dict

    Returns
    -------
    str

    """
    args = ['='.join([key, str(value)]) for key, value in d.items()]
    return '&'.join(args)


def is_long_nan(v):
    if v == long_nan:
        return True
    else:
        return False


def to_nan(x):
    if is_long_nan(x):
        return np.nan
    else:
        return x


def _to_date(row):
    date = int(row['DATE'])
    return pd.datetime(year=date // 10000, month=date // 100 % 100, day=date % 100)


def _to_datetime(row):
    date = int(row['DATE'])
    time = int(row['TIME']) // 1000
    return pd.datetime(year=date // 10000, month=date // 100 % 100, day=date % 100,
                       hour=time // 10000, minute=time // 100 % 100, second=time % 100)


def to_dataframe(cloumset, index_func=None, index_column=None):
    df = pd.DataFrame(cloumset)
    for col in df.columns:
        if df.dtypes.loc[col] == np.int64:
            df.loc[:, col] = df.loc[:, col].apply(to_nan)
    if index_func:
        df.index = df.apply(index_func, axis=1)
    elif index_column:
        df.index = df[index_column]
        del df.index.name

    return df


def _error_to_str(error):
    if error:
        if 'message' in error:
            return str(error['error']) + "," + error['message']
        else:
            return str(error['error']) + ","
    else:
        return ","


def to_obj(class_name, data):
    try:
        if isinstance(data, (list, tuple)):
            result = []
            for d in data:
                result.append(namedtuple(class_name, list(d.keys()))(*list(d.values())))
            return result

        elif type(data) == dict:
            result = namedtuple(class_name, list(data.keys()))(*list(data.values()))
            return result
        else:
            return data
    except Exception as e:
        print(class_name, data, e)
        return data


def to_date_int(date):
    if isinstance(date, str):
        tmp = date.replace('-', '')
        return int(tmp)
    elif isinstance(date, (int, np.integer)):
        return date
    else:
        return -1


def to_time_int(time):
    if isinstance(time, str):
        if ':' in time:
            tmp = time.replace(':', '')
            return int(tmp)
        else:
            t = dt.datetime.strptime(time, "%H:%M:%S")
            time_int = t.hour * 10000 + t.minute * 100 + t.second
            return time_int

    elif isinstance(time, (int, np.integer)):
        return time
    else:
        return -1


def get_mtype(symbol):
    if re.match('\d{6}\.\w+', symbol):
        return 'CS'
    else:
        return 'Future'


def get_atype(atype):
    return 'none' if atype is None else atype
