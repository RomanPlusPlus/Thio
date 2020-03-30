""" Converts texts to datapoints, and the other way around.
Will be made mostly obsolete after the planned transition to a database is done.
"""

import pandas as pd
import helper_funcs
import utils

channels = utils.read_configs()["data_channels"]


def find_all_positions_of_character(istr, character):
    return [pos for pos, char in enumerate(istr) if char == character]  # a list


def get_price_from_substr(istr, original_str):
    """Extracts the float value (e.g. 7.5) and the channel_name from a string like this: "channel_name; unit; 7.5".

    Args:
        istr (str): a string like this: "bitcoin; eur; 7.5"
        original_str (str): at the very first stage of parcing, before this func is called, we receive a string like:
            1585474566.27; bitcoin; eur; 3.664121010741326 § ethereum; eur; 1.0710547987175814 ...
            We pass it here for debug purposes.
    """

    positions_list = find_all_positions_of_character(istr, ';')
    if len(positions_list) > 0:
        temp_list = positions_list[-1:]  # get the position of last ";" as a list of 1 element
    else:
        temp_list = []
    if len(temp_list) > 0:
        position = temp_list[0]
        position += 2  # skip "; "
        price_str = istr[position:]  # get the string from this position
        if ("None" in price_str) or ("invalid" in price_str):
            price = -1
            helper_funcs.append_logs(
                "get_price_from_substr: -None- or -invalid- in the input string. Could be just a missing data, "
                "or a sign of something bad. Input: " + str(
                    istr) + " . Original str: " + original_str, "parser", "always")
        else:
            try:
                price = float(price_str)  # try to convert it into float
            except Exception as e:
                price = -1  # if can't parse the price, return "-1"  
                msg = "ERROR: get_price_from_substr: price = float(price_str) caused an arror: " + str(
                    e) + " . Inputs: istr = " + str(istr)
                helper_funcs.append_logs(msg, "parser", "always", "print")

        position = positions_list[0]  # get the position of the first ";" to remove the timestamp
        name_str = istr[:position]

    else:
        price = -1
        name_str = ""
        msg = "get_price_from_substr: len(temp_list) is zero. Caused by this istr: " + istr
        helper_funcs.append_logs(msg, "parser", "always", "print")

    return price, name_str


def split_price_str_into_substrings(istr):
    """ Returns a list of strings, where each string looks like this: "bitcoin; eur; 8080.99"

    Args:
        istr (str): a string like this: 1582830400.15; bitcoin; eur; 8080.99 § litecoin; eur; 58.08
        """
    temp_list = find_all_positions_of_character(istr, ';')
    if len(temp_list) > 0:
        position = temp_list[0]  # get the position of the first ";" to remove the timestamp
        position += 2  # skip "; "
        istr_without_time = istr[position:]
        res = istr_without_time.split(" § ")
    else:
        res = ["-1"]
    return res


def extract_timestamp(istr):
    """ Returns a unix timestamp as a float (e.g. 1582830400.15).

    Args:
        istr (str): a string like this: 1582830400.15; bitcoin; eur; 8080.99 § litecoin; eur; 58.08
        """
    temp_list = find_all_positions_of_character(istr, ';')
    if len(temp_list) > 0:
        position = temp_list[0]  # get the position of the first ";"
        stamp_str = istr[:position]
        res = float(stamp_str)
    else:
        res = 0
    return res


def get_prices_from_string(istr):
    """Returns a dic in the form {"bitcoin" : 8080.99, ....}, and a unix timestamp as a float (e.g. 1582830400.15)

    Args:
        istr (str): a string like this: 1582830400.15; bitcoin; eur; 8080.99 § litecoin; eur; 58.08
        """

    timestamp = extract_timestamp(istr)
    substrings_list = split_price_str_into_substrings(istr)
    prices_dic = dict()
    for gpf in range(len(substrings_list)):
        price, name_str = get_price_from_substr(substrings_list[gpf], istr)
        prices_dic[name_str] = price

    return prices_dic, timestamp


def dic_keys_sorted_list(idic):
    """Converts dictionary keys into an alphabetically sorted list of strings.

    Args:
        idic (dict): a dict whose keys are strings
        """

    # TODO: replace -1 with None, both here and the caller func
    if idic != -1:
        names = []
        for key, value in idic.items():
            names.append(key)
        res = sorted(names)
    else:
        res = -1
    return res


def price_dict_to_str(price_dic, measure_unit, unixtime):
    """ Returns a string that looks like this: 1582830400.15; bitcoin; eur; 8080.99 § litecoin; eur; 58.08

    Args:
       price_dic (dict): a dict like this: {"bitcoin": 8080.99, ....}
       measure_unit (str): a string that indicates the unit of measurement (e.g. "eur")
       unixtime (float): a unix timestamp (e.g. 1582830400.15)
    """

    res = str(round(unixtime, 2)) + "; "
    names = dic_keys_sorted_list(price_dic)  # to ensure that channels are always listed in the same order
    if names != -1:
        for pdt in range(len(names)):
            res += names[pdt] + "; "
            res += measure_unit + "; "
            res += str(price_dic[names[pdt]]) + " § "
        res = res[:-3]  # to remove the last " § "
    else:
        res = "-1"
    return res


def fetched_data_to_dataframe(filename):
    """ Reads the dataset file and converts it into a pandas dataframe, with columns representing channels.

    Args:
       filename (str): the dataset filename (e.g. "syntheticData.txt")
    """
    cols_number = 3 * len(channels)

    my_cols = [str(i) for i in range(cols_number)]  # create some row names

    cols2delete = []
    for c in range(cols_number):
        if (c + 1) % 3 != 0:
            cols2delete.append(c)

    df = pd.DataFrame()
    try:
        # TODO: use isNonZeroFile7 to check if non zero
        df = pd.read_csv(helper_funcs.get_full_path(filename),
                         sep=";|§",
                         names=my_cols,
                         header=None,
                         engine="python")
    except Exception as e:
        helper_funcs.append_logs("ERROR in fetchedData_to_DataFrame upon trying to open " + filename + " : " + str(e),
                                 "parser", "always", "print")

    if not df.empty:
        df = df.drop(df.columns[cols2delete], axis=1)

        # the columns in the fetched file are sorted alphabetically.
        # We sort it here too - to make them be the same columns
        df.columns = sorted(channels)
    return df
