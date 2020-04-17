""" Creates/updates the Telemanom model, by regularly training it on the latest N datapoints.

It runs in parallel with the lib_telemanom_infer.py, to make training and inference work as separate processes.
"""

from helper_funcs import append_logs, train_and_save_model, exit7
import lib_telemanom_calc

get_model_func = lib_telemanom_calc.get_model
name4logs = "lib_telemanom_train"
method_name = "telemanom"
use_this_many_latest_dp = 30000  # bigger number means better AI but more compute required

# main circle
append_logs("starting the training circling", name4logs, "always", "print")
while True:
    exit7()
    train_and_save_model(use_this_many_latest_dp, get_model_func, method_name)
