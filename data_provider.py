""" Provides new datapoints, both real-life and synthetic.

The real-life data are cryptocurency exchange rates, acquired through the CoinGecko API.
The synthetic data are generated on the fly. If you plot it, it kinda looks like data fetched from physical sensors.
"""

import time
import math
import random
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from helper_funcs import is_nonzero_file7, get_full_path, append_logs, list_to_file
from parser import price_dict_to_str

summon_filename = "state_controls/summonAnomaly.txt"
name4logs = "data_provider"


# ======================== real-world data (crypto exchange rates) ==========================

# source: https://github.com/man-c/pycoingecko
def api_url_params(api_url, params):
    if params:
        api_url += '?'
        for key, value in params.items():
            api_url += "{0}={1}&".format(key, value)
        api_url = api_url[:-1]
    return api_url


class CoinGeckoAPI:
    """ Provides current crypto exchange rates. It's an abridged version of https://github.com/man-c/pycoingecko"""

    __API_URL_BASE = 'https://api.coingecko.com/api/v3/'

    def __init__(self, api_base_url=__API_URL_BASE):
        self.api_base_url = api_base_url
        self.request_timeout = 120

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def __request(self, url):
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            content = json.loads(response.content.decode('utf-8'))
            return content
        except Exception as e:
            raise

    def get_price(self, ids, vs_currencies, **kwargs):
        """Get the current price of any cryptocurrencies in any other supported currencies that you need"""

        # remove empty spaces (when querying more than 1 coin, comma-separated,
        # spaces may exist between coins ie ids='bitcoin, litecoin' -> ids='bitcoin,litecoin')
        ids = ids.replace(' ', '')
        kwargs['ids'] = ids
        vs_currencies = vs_currencies.replace(' ', '')
        kwargs['vs_currencies'] = vs_currencies

        api_url = '{0}simple/price'.format(self.api_base_url)
        api_url = api_url_params(api_url, kwargs)

        return self.__request(api_url)


def fetched_datapoint(names):
    """ Fetches crypto exchange rates, returns a dict of prices and a unix timestamp.

    For example, the output could look like this:
    ({'bitcoin': 6108.71, 'monero': 41.95}, 1585066030.1136858)

    Args:
        names (list of strings): names of cryptos. E.g. ["bitcoin", "monero"]
    """
    measuring_unit = "eur"

    ids_str = ""
    for i in range(len(names)):
        ids_str += names[i] + ","
    ids_str = ids_str[:-1]

    cg = CoinGeckoAPI()
    data_point_raw = cg.get_price(ids=ids_str, vs_currencies=measuring_unit)
    timestamp = time.time()

    prices_dict = dict()
    for sd in range(len(names)):
        value = data_point_raw[names[sd]][measuring_unit]
        prices_dict[names[sd]] = value

    return prices_dict, timestamp


# ======================== synthetic data ==========================


def user_requested_anomaly7():
    """ Checks if the user requested an anomaly, and returns True/False accordingly. """
    digit = 0
    res = False
    if is_nonzero_file7(summon_filename):
        lines = []
        with open(get_full_path(summon_filename)) as f:
            lines = f.readlines()
        if len(lines) > 0:
            try:
                digit = int(lines[0])
                if digit > 0:
                    res = True
            except Exception as e:
                res = False
                append_logs("ERROR:" + str(e), name4logs, "always")
        else:
            res = False
    else:
        res = False

    # Disable summoning of anomalies after the requested number of anomalies were added
    if res:
        with open(get_full_path(summon_filename), "w") as f:
            if digit > 0:
                f.write(str(digit - 1))
            else:
                f.write("0")

    return res


def add_anomaly(datapoint):
    if user_requested_anomaly7():
        prices_dict, timestamp = datapoint
        for key, value in prices_dict.items():
            prices_dict[key] = prices_dict[key] * 2
        res = prices_dict, timestamp
    else:
        res = datapoint
    return res


def add_noise(normal_value):
    magnitude = random.randint(-6, 6) / 100
    return normal_value + normal_value * magnitude


def synthetic_datapoint(names):
    """ Generates a synthetic datapoint, returns a dict of values and a unix timestamp.

    Designed in such a way as to produce a cool-looking curve if you plot the consecutive points.
    For each channel (defined in -names-), there will be a differently-looking plot.
    Args:
        names(list of strings): names of channels
    """

    start_time = 1583673336.2082493

    timestamp = time.time()

    prices_dict = dict()
    for sd in range(len(names)):
        slowdown_factor = 2 ** sd + 1

        argument = timestamp - start_time
        argument = (argument + sd * math.sin(argument)) // slowdown_factor
        value = (sd + 1) * math.sin(argument) + sd + 2

        value = add_noise(value)
        prices_dict[names[sd]] = value

    return prices_dict, timestamp


# ======================== unified data provider ==========================

def recieve_datapoint(names, synthetic7):
    if synthetic7:
        res = synthetic_datapoint(names)
    else:
        res = fetched_datapoint(names)
    res = add_anomaly(res)
    return res


def fetch_and_save_datapoint(data_channels, use_synthetic_data7):
    """ Returns a string that looks like this: 1582830400.15; bitcoin; eur; 8080.99 § litecoin; eur; 58.08

    Also saves the string to the latest_datapoint file.
    Args:
        data_channels (list of strings): names of channels
        use_synthetic_data7 (bool): True if synthetic data is used, False otherwise
    """
    try:
        price_dic, ts = recieve_datapoint(data_channels, use_synthetic_data7)
        data_point_str = price_dict_to_str(price_dic, "eur", ts)

        if use_synthetic_data7:
            data_filename_for_saving = "dataset/latest_datapoint_synthetic.txt"
        else:
            data_filename_for_saving = "dataset/latest_datapoint_fetched.txt"

        # TODO: move it to 0launcher
        list_to_file(data_filename_for_saving, [data_point_str], "w")
    except Exception as fetch_e:
        data_point_str = None
        f_msg = "failed to get a datapoint: " + str(fetch_e)
        append_logs(f_msg, "0launcher", "always", "print")
    return data_point_str


# ----------------------------------

# Parts of the code in this file were adapted from the code written by Christoforou Emmanouil, 2018
# that was released under the MIT License (see below).
# source: https://github.com/man-c/pycoingecko
#
# MIT License
#
# Copyright (c) 2018 Christoforou Emmanouil
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
