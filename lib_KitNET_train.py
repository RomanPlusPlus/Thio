""" Creates/updates the KitNET model, by regularly training it on the latest N datapoints.

It runs in parallel with the inference script (lib_KitNET_infer.py), supplying it with new models to infer upon.

Note: if you want a model that produce meaningful results, you need a dataset of at least 50 000 datapoints.
"""


from helper_funcs import append_logs, train_and_save_model, exit7
import lib_KitNET_calc

get_model_func = lib_KitNET_calc.get_model
name4logs = "lib_KitNET_train"
method_name = "kitnet"
use_this_many_latest_dp = 100000  # bigger number means better AI but more compute required

# main circle
append_logs("starting the training circling", name4logs, "always", "print")
while True:
    exit7()
    train_and_save_model(use_this_many_latest_dp, get_model_func, method_name)
