""" A collection of tools to generate self-diagnostics info to provide it to the GUI.
"""

import psutil
import time

from helper_funcs import append_logs, get_file_modification_ts, synthetic_data7

name4logs = "self_diagnostics"

use_synthetic_data7 = synthetic_data7()
if use_synthetic_data7:
    output_postfix = "_synthetic"
else:
    output_postfix = "_fetched"

# TODO: move it to configs
methods = ["telemanom", "kitnet"]


def python_script_running7(script_filename):
    """ Returns True if the script with the given filename is currently running, False otherwise.

    Args:
        script_filename (str): e.g. "lib_telemanom_train.py"
    """
    res = False
    try:
        for p in psutil.process_iter():
            if len(p.cmdline()) > 1:
                if script_filename in p.cmdline()[1]:
                    res = True
                    break
    except Exception as e:
        append_logs("Exception: " + str(e), name4logs, "always")
    return res


def get_dead_scripts():
    """ Returns a list of scripts that should be running, but are dead instead.
    """
    # TODO: load it only once
    with open("state_controls/scripts_to_run.txt") as scripts_f:
        all_filenames = scripts_f.readlines()
    dead_scripts = []
    for name in all_filenames:
        if not python_script_running7(name.strip()):
            dead_scripts.append(name.strip())
    return dead_scripts


def get_models_modification_ts(method_name):
    """ Returns the UNIX epoch timestamp of the modification time of the given model.

    Args:
        method_name (str): e.g. "telemanom"
    """
    modelpath = "pickled_models/" + method_name + output_postfix + ".pkl"
    model_ts = get_file_modification_ts(modelpath)
    return model_ts


def get_wenn_models_were_updated_string():
    """ Generates a human-readable str about when models were last modified.

    The result looks like this: "When the models were retrained:\n Telemamom: 3 min ago"
    """

    now = time.time()

    status_str = "When the models were retrained:\n"
    for m in methods:
        modification_ts = get_models_modification_ts(m)
        if modification_ts is not None:
            timestamps_diff = now - modification_ts
            mins_str = str(round(timestamps_diff / 60))
            status_str += m + ": " + mins_str + " min ago\n"
        else:
            status_str += m + ": doesn't exist yet. First launch?\n"
    status_str = status_str[:-1]

    return status_str
