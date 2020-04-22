""" It's the script you should execute to launch the whole thing.

Make sure to create the conda environments named thio_telemanom and thio_kitnet with all the dependencies installed, as
described in the readme.

The most important things this script does are:
- resetting states (deletes previous logs etc)
- preprocessing the dataset
- launching other scripts for parallel execution of training, inference, gui etc
- regularly fetching and saving new datapoints
"""

import time

from helper_funcs import get_full_path, delete_logs, append_logs, list_to_file, exit7, synthetic_data7
from launch_utils import read_configs, launch_scripts
import data_provider
from dataset_preprocessing import cleanup_dataset, data_sanity_check

time_between_fetches = 1.0  # how often should the data be fetched from the data provider, in seconds
this_many_last_observations = 500  # to save them TO a separate file

# Reset states
delete_logs()  # delete logs from the previous sessions
with open(get_full_path("state_controls/summonAnomaly.txt"), "w") as f:
    f.write("0")
with open(get_full_path("state_controls/exit7.txt"), "w") as f:
    f.write("0")

data_channels = read_configs()["data_channels"]

use_synthetic_data7 = synthetic_data7()
append_logs("use_synthetic_data7 : " + str(use_synthetic_data7), "0launcher", "always", "print")
append_logs("data_channels : " + str(data_channels), "0launcher", "always", "print")

cleanup_dataset(use_synthetic_data7)
data_sanity_check(use_synthetic_data7, data_channels)

launch_scripts()

if use_synthetic_data7:
    data_filename = "dataset/syntheticData.txt"
    last_n_filename = "dataset/lastNpoints_synthetic.txt"
else:
    data_filename = "dataset/fetchedData.txt"
    last_n_filename = "dataset/lastNpoints_fetched.txt"

latest_datapoints = []
while True:
    exit7()

    datapoint = data_provider.fetch_and_save_datapoint(data_channels, use_synthetic_data7)

    if datapoint is not None:

        latest_datapoints.append(datapoint)
        if len(latest_datapoints) > this_many_last_observations:
            tempArr = latest_datapoints[-this_many_last_observations:]
            latest_datapoints = tempArr

        list_to_file(data_filename, [str(datapoint) + "\n"], "a")

        with open(get_full_path(last_n_filename), "w") as f:
            for s in latest_datapoints:
                f.write(str(s) + "\n")
            f.flush()
        f.close()
        print(datapoint)
        print("")
    else:
        print("Failed to fetch a new datapoint. Will wait 10 sec and try again")
        time.sleep(10)

    time.sleep(time_between_fetches)
