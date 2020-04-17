""" Contains a diverse set of helper functions, from working with files to the training infrastructure.

TODO: split thematically related funcs into separate modules for better maintainability
"""

import os
import time
import datetime
import pickle
import pandas as pd
import numpy as np

import parser
import launch_utils

logs_fname = "logs/logs.txt"
name4logs = "helper_funcs"


def synthetic_data7():
    """ Reads configs and returns True if configs say that synthetic data should be used, and False otherwise
    """
    config = launch_utils.read_configs()
    if config["data_source"] == "synthetic":
        res = True
    else:
        res = False
    return res


use_synthetic_data7 = synthetic_data7()
cryptos = launch_utils.read_configs()["data_channels"]
log_verbosity = launch_utils.read_configs()["log_verbosity"]


def get_full_path(filename):
    """ Returns the full path of a file assuming that it's located in the same dir as this script.

    Args:
        filename (str): the name of the file, including the extension
    """
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    full_path = os.path.join(__location__, filename)
    return full_path


def is_file7(fpath):
    return os.path.isfile(get_full_path(fpath))


def is_nonzero_file7(fpath):
    if is_file7(fpath):
        if os.path.getsize(get_full_path(fpath)) > 0:
            res = True
        else:
            res = False
    else:
        res = False
    return res


def exit7():
    """ Checks if the user requested termination of all Thio scripts. If yes, exits. Called from each script. """

    exit_filename = "state_controls/exit7.txt"
    if is_nonzero_file7(exit_filename):
        with open(get_full_path(exit_filename)) as f:
            lines = f.readlines()
        if len(lines) > 0:
            try:
                digit = int(lines[0])
                if digit == 1:
                    exit()
            except Exception as e:
                append_logs("ERROR: " + str(e), name4logs, "always")


def get_file_modification_ts(fpath):
    if is_file7(fpath):
        res = os.path.getmtime(get_full_path(fpath))
    else:
        append_logs("get_file_modification_ts : no such file as " + str(fpath), "helper_funcs", "always")
        res = None
    return res


def human_readable_timestamp():
    now = datetime.datetime.now()
    time_st = now.strftime('%Y-%m-%d %H:%M:%S')
    return time_st


def worthy_log7(verbosity, entry_mode):
    res = False
    if verbosity == "show_all":
        res = True
    if (verbosity == "show_normal_and_higher") and (entry_mode in ["normal", "always"]):
        res = True
    if (verbosity == "show_only_critical") and (entry_mode == "always"):
        res = True
    return res


# entry_mode can be "verbose", "normal", "always" 
def append_logs(istr, func_name, entry_mode, print7str=""):
    if worthy_log7(log_verbosity, entry_mode):
        msg = human_readable_timestamp() + " : " + func_name + " : " + istr + "\n"
        if print7str == "print":
            print(msg)
        with open(get_full_path(logs_fname), "a") as myFile:
            myFile.write(msg)
            myFile.flush()
        myFile.close()


def delete_logs():
    with open(get_full_path(logs_fname), "w") as myFile:
        myFile.write("")


def latest_prices_for_this_crypto(latest_price_dicts_list, name):
    """ Returns a list of prices (floats).

    Args:
        latest_price_dicts_list (a list of dicts): each dict looks like this: {'channel_name': value,...}
        name (str): the name of the crypto, for which the latest prices must be returned
    """
    latest_prices = []
    for lpf in range(len(latest_price_dicts_list)):
        single_dict, timestamp = latest_price_dicts_list[lpf]
        if name in single_dict:
            price_point = single_dict[name]
            if price_point != -1:
                latest_prices.append(price_point)
    return latest_prices


def read_latest_points(filename, use_only_this_many_latest):
    try:
        if is_file7(filename):
            with open(get_full_path(filename)) as f:
                lines = f.readlines()
        else:
            lines = []
    except Exception as e:
        lines = []
        msg = "Exception while trying to open " + filename + " . Exception: " + str(e) + " . The user deleted it?"
        append_logs(msg, "helper_funcs", "always", "print")
    if len(lines) > use_only_this_many_latest:
        lines = lines[-use_only_this_many_latest:]
    return lines


# return a list, where each element is a tuple (prices_dict, timestamp)
def price_dicts_list(lines):
    latest_price_dicts_list = []
    for i in range(len(lines)):
        try:
            prices_dict, timestamp = parser.get_prices_from_string(lines[i])
            latest_price_dicts_list.append((prices_dict, timestamp))
        except Exception as e:
            append_logs("Price_dicts_list : " + str(e), "helper_funcs", "always")
    return latest_price_dicts_list


def read_prediction_for_aggregation(filename, who_asking=""):
    if who_asking != "":
        who_asked_str = " . who asked: " + who_asking
    else:
        who_asked_str = ""
    try:
        if is_file7(filename):
            with open(get_full_path(filename)) as f:
                lines = f.readlines()
        else:
            lines = []
    except Exception as e:
        lines = []
        append_logs("read_prediction_for_aggregation: Unable to open " + filename + " : " + str(
            e) + who_asked_str + " . If it's the first launch without any data, this error usually can be ignored",
                    "helper_funcs", "always", "print")
    if len(lines) > 0:
        predicted_dic, timestamp = parser.get_prices_from_string(lines[0])
        return predicted_dic
    else:
        msg = "read_prediction_for_aggregation: " + filename + "Seems to be empty or non-existant. The user deleted " \
                                                               "it?." + who_asked_str + " . If it's the first launch " \
                                                                                        "without any data, " \
                                                                                        "this error usually can be " \
                                                                                        "ignored "
        append_logs(msg, "helper_funcs", "always", "print")
        return None


# I'm a pickle, Morty!
def save_model_to_pickle(model, scales_dic, timestamp, filename):
    try:
        obj_to_save = (model, scales_dic, timestamp)
        with open(get_full_path(filename), 'wb') as pkl:
            pickle.dump(obj_to_save, pkl)
    except Exception as e:
        msg = "failed to save model to pickle " + str(filename) + " . Exception: " + str(e)
        append_logs(msg, "helper_funcs", "always", "print")


def read_model_from_pickle(filename):
    try:
        if is_nonzero_file7(filename):
            with open(get_full_path(filename), 'rb') as pkl:
                model, scales_dic, timestamp = pickle.load(pkl)
            append_logs("loaded model and timestamp " + str(timestamp), "helper_funcs", "verbose", "print")
        else:
            model = None
            timestamp = None
            scales_dic = None

    except Exception as e:
        model = None
        timestamp = None
        scales_dic = None
        msg = "failed to read model from pickle " + str(filename) + " .Exception:" + str(e)
        append_logs(msg, "helper_funcs", "always", "print")
    return model, scales_dic, timestamp


def read_last_lines(filename, window=1):
    """ Reads the N last lines of a file and output the list of them. Without loading the entire file into memory.

    It's a modified version of this code by Hauke Rehfeld, 2018: https://stackoverflow.com/a/48087596

    Args:
        filename (str): the name of the file to read from
        window (int): how many last raws to read
    """
    if is_nonzero_file7(filename):
        with open(get_full_path(filename), 'rb') as f:
            # Get the last `window` lines of file `f` as a list of bytes.
            if window == 0:
                return b''
            bufsize = 1024
            f.seek(0, 2)
            end = f.tell()
            nlines = window + 1
            data = []
            while nlines > 0 and end > 0:
                i = max(0, end - bufsize)
                nread = min(end, bufsize)
                f.seek(i)
                chunk = f.read(nread)
                data.append(chunk)
                nlines -= chunk.count(b'\n')
                end -= nread
            list_of_bytes = b'\n'.join(b''.join(reversed(data)).splitlines()[-window:])

            # convert it to utf-8
            decoded_text = list_of_bytes.decode('utf-8')
            output_list = decoded_text.split('\n')
    else:
        output_list = []

    return output_list


def anomaly_str_to_ts_and_score(istr):
    """ Parses a string that contains a timestamp and an anomaly score.

    Output looks like this (a tuple of floats):
    (1584452172.12, 0.0071348165601308475)

    Args:
        istr(str): looks like this: 1584452172.12; 0.0071348165601308475
    """
    semicolons_ind = istr.find(';')
    if semicolons_ind > 0:
        stamp_str = istr[:semicolons_ind]
        score_str = istr[semicolons_ind + 1:]
        timestamp = float(stamp_str)
        score = float(score_str)
        res = timestamp, score
    else:
        res = None, None
    return res


def get_latest_anomaly_scores(filename, scores_num):
    """ Reads the file that contains anomaly scores, and returns them as a list of floats.

    Args:
        filename (str): each row in the file looks like this (timestamp, score):
            1584452170.96; 0.007139212893789588
        scores_num (int): how many scores to read
    """
    lines = read_last_lines(filename, scores_num)
    olist = []
    for gla in range(len(lines)):
        ts_and_score_touple = anomaly_str_to_ts_and_score(lines[gla])
        olist.append(ts_and_score_touple)
    return olist


def get_max_anomaly_from_latest(filename, scores_num):
    ts_and_score_list = get_latest_anomaly_scores(filename, scores_num)
    scores = []
    for gma in range(len(ts_and_score_list)):
        ts, score = ts_and_score_list[gma]
        scores.append(score)
    if len(scores) > 0:
        res = max(scores)
    else:
        res = 0
    return res


def list_to_file(filename, ilist, write_mode):
    with open(get_full_path(filename), write_mode) as myFile:
        for s in ilist:
            myFile.write(s)
        myFile.flush()
    myFile.close()


def train_and_save_model(use_only_this_many_latest, get_model_func, method_name):
    if use_synthetic_data7:
        dataset_filename = "dataset/syntheticData.txt"
    else:
        dataset_filename = "dataset/fetchedData.txt"

    input_dataframe = parser.fetched_data_to_dataframe(dataset_filename, use_only_this_many_latest)

    if not input_dataframe.empty:
        timestamps = pd.DataFrame(input_dataframe.index).to_numpy()
        latest_timestamp = timestamps[-1]
        datapoints_number = len(input_dataframe.index)
    else:
        latest_timestamp = None
        datapoints_number = 0

    append_logs(
        "Got this many datapoints: " + str(datapoints_number) + " . latest_timestamp = " + str(latest_timestamp),
        name4logs, "verbose")

    if datapoints_number > 0:
        try:
            metamodel, scales_dic, success = get_model_func(input_dataframe)

            if not success:  # There is likely not enough data to build models. Sleep a bit
                time.sleep(120)
                append_logs("There is likely not enough data to build models. Sleep a bit.", name4logs, "always",
                            "print")

        except Exception as e:
            metamodel = None
            scales_dic = None
            append_logs("ERROR: failed to to get metamodel: " + str(e), name4logs, "always", "print")

        if use_synthetic_data7:
            output_postfix = "_synthetic"
        else:
            output_postfix = "_fetched"

        save_model_to_pickle(metamodel, scales_dic, latest_timestamp,
                             "pickled_models/" + method_name + output_postfix + ".pkl")

        append_logs("Trained and saved a new model for " + method_name + ", using this many datapoints: " + str(
            datapoints_number), name4logs, "normal")


def obtain_model(modelpath, old_modification_ts, old_meta_model_dic, old_scales_dic):
    new_modification_ts = get_file_modification_ts(modelpath)
    if int(old_modification_ts) < int(
            new_modification_ts):  # to avoid loading old models again, as models can be massive
        append_logs("The model was updated. Prev model's file was modified at " + str(
            old_modification_ts) + " . This model's file was modified at " + str(new_modification_ts) + " . Loading it",
                    name4logs, "verbose")
        new_meta_model_dic, new_scales_dic, model_ts = read_model_from_pickle(modelpath)

        old_meta_model_dic = new_meta_model_dic
        old_modification_ts = new_modification_ts
        old_scales_dic = new_scales_dic

    else:
        new_meta_model_dic = old_meta_model_dic
        model_ts = None
        new_scales_dic = old_scales_dic

    return old_meta_model_dic, new_meta_model_dic, model_ts, old_modification_ts, old_scales_dic, new_scales_dic


def infer_and_save_results(ask_model_func, modelpath, old_modification_ts, old_meta_model_dic,
                           use_only_this_many_latest, method_name, scales_dic):
    if use_synthetic_data7:
        latest_points_filename = "dataset/lastNpoints_synthetic.txt"
        output_postfix = "_synthetic"
    else:
        latest_points_filename = "dataset/lastNpoints_fetched.txt"
        output_postfix = "_fetched"

    if is_file7(modelpath):

        old_meta_model_dic, new_meta_model_dic, model_ts, old_modification_ts, old_scales_dic, new_scales_dic = obtain_model(
            modelpath,
            old_modification_ts,
            old_meta_model_dic,
            scales_dic)

        append_logs("Obtained model for " + method_name + ": new_scales_dic: " + str(new_scales_dic), name4logs,
                    "verbose")

        observations_df = parser.fetched_data_to_dataframe(latest_points_filename, use_only_this_many_latest)

        if observations_df.shape[0] > 0:
            try:
                anomaly_score = ask_model_func(new_meta_model_dic, observations_df, new_scales_dic)
            except Exception as e:
                anomaly_score = 0
                append_logs("ERROR: An attempt to ask a model for " + method_name + " caused an exception: " + str(e),
                            name4logs, "always")

            anomaly_ts = time.time()

            str2write = str(round(anomaly_ts, 2)) + "; " + str(anomaly_score) + "\n"

            save_path = get_full_path("risk_scores/" + method_name + "_anomaly" + output_postfix + ".txt")
            with open(save_path, "a") as myFile:
                myFile.write(str2write)

    else:
        append_logs(
            "Can't load the " + method_name + " model because it doesn't exist yet. Will wait for 30 sec and try again",
            name4logs, "always")
        time.sleep(30)

    return old_modification_ts, old_meta_model_dic, scales_dic


def get_scaling_factors(input_arr):
    """ Calculates the 0.25 and 0.75 quantiles, and then expands them for a safety margin

    Args:
        input_arr (a list of floats OR a single-column dataframe): should be of a non-zero len 

    """
    if isinstance(input_arr, list):
        df = pd.DataFrame(input_arr)
        lower_quantile, upper_quantile = df[0].quantile([.25, .75])
    else:
        col_name = list(input_arr.columns.values)[0]
        lower_quantile, upper_quantile = input_arr[col_name].quantile([.25, .75])

    # How we ensure that a safety margin is added properly:
    # Let's say the quantile vales are 2 and 6.
    # In this case, min_val and max_val are 1 and 12.
    # For quantile vales of -2 and 6, they should be -4 and 12
    # For quantile vales of -6 and -2, they should be -12 and -1
    min_val = lower_quantile * 0.5
    max_val = upper_quantile * 2.0
    if min_val < 0:
        min_val *= 4.0
    if max_val < 0:
        max_val *= 0.25

    return min_val, max_val
