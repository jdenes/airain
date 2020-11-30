import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def week_of_month(dt):
    """
    Get which week of a month a day belongs to.

    :param datetime dt: the day you want to know which week of month it belongs to.
    :return: week of the month number.
    :rtype: int
    """

    dt = pd.to_datetime(dt)
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom/7))


def next_day(date):
    """
    Returns next banking day.

    :param str date: the day you want to know which is the next banking day, in format '%Y-%m-%d'.
    :return: previous banking day in format '%Y-%m-%d'.
    :rtype: str
    """

    date = datetime.strptime(date, '%Y-%m-%d')
    if date.weekday() == 4:
        res = date + timedelta(days=3)
    elif date.weekday() == 5:
        res = date + timedelta(days=2)
    else:
        res = date + timedelta(days=1)
    return res.strftime('%Y-%m-%d')


def previous_day(date):
    """
    Returns previous banking day.

    :param str date: the day you want to know which is the previous banking day, in format '%Y-%m-%d'.
    :return: previous banking day in format '%Y-%m-%d'.
    :rtype: str
    """

    date = datetime.strptime(date, '%Y-%m-%d')
    if date.weekday() == 0:
        res = date - timedelta(days=3)
    elif date.weekday() == 6:
        res = date - timedelta(days=2)
    else:
        res = date - timedelta(days=1)
    return res.strftime('%Y-%m-%d')
