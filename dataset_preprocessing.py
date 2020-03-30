""" Prepares the dataset for training, by cleaning it and checking its integrity."""

import pandas as pd

from helper_funcs import is_nonzero_file7, get_full_path, append_logs

name4logs = "dataset_preprocessing"


# TODO: make a backup first
def cleanup_data(filename):
    """ Removes bad strings from the dataset file.

    Args:
       filename (str): the dataset filename (e.g. "syntheticData.txt")
    """
    badfile = False
    lines = []
    if is_nonzero_file7(filename):
        with open(get_full_path(filename)) as f:
            lines = f.readlines()
    else:
        badfile = True

    if not badfile:
        cleaned = []
        for cfd in range(len(lines)):
            if ("invalid" not in lines[cfd]) and ("None" not in lines[cfd]):
                cleaned.append(lines[cfd])

        with open(get_full_path(filename), "w") as myFile:
            for cfd in range(len(cleaned)):
                myFile.write(cleaned[cfd])


def cleanup_dataset(using_synthetic_data7):
    if using_synthetic_data7:
        cleanup_data("dataset/syntheticData.txt")
        cleanup_data("lastNpoints_synthetic")
        cleanup_data("latest_datapoint_synthetic")

    else:
        cleanup_data("dataset/fetchedData.txt")
        cleanup_data("lastNpoints_fetched")
        cleanup_data("latest_datapoint_fetched")


def data_sanity_check(use_synthetic_data7, data_channels):
    """ Checks if the dataset contains corrupted data.

    The checks cover the case where the user has changed the number of channels, but forgot to delete the old data that
    still has the old number of channels.

    Args:
        use_synthetic_data7 (bool): if True, the synthetic data is used
        data_channels (list of strings): channel names

    """
    # TODO: check if there are at least 2 channels in configs, otherwise KitNET will not work

    if use_synthetic_data7:
        dataset_filename = "dataset/syntheticData.txt"
    else:
        dataset_filename = "dataset/fetchedData.txt"

    bad_shape_msg = dataset_filename + "seems to be in a bad shape, as reading it into a dataframe causes an error" \
                                       " or meaningless output. If you changed the number of channels, deleting   " \
                                       "the data that has the previous number of channels could help "

    # TODO: remove code duplication, as a similar code is used in fetched_data_to_dataframe
    if is_nonzero_file7(dataset_filename):
        cols_number = 3 * len(data_channels)
        my_cols = [str(i) for i in range(cols_number)]  # create some row names
        print("checking...", dataset_filename)
        df = pd.DataFrame()
        try:
            df = pd.read_csv(get_full_path(dataset_filename),
                             sep=";|§",
                             names=my_cols,
                             header=None,
                             engine="python")
        except Exception as e:
            append_logs(bad_shape_msg + " " + str(e), name4logs, "always", "print")
            exit()

        timestamps = pd.DataFrame(df.index).to_numpy()
        latest_timestamp = timestamps[-1]
        if "nan" in str(latest_timestamp):
            append_logs(bad_shape_msg, name4logs, "always", "print")
            exit()
    else:
        append_logs(dataset_filename + " doesn't exist or of zero size. First launch?", name4logs, "always", "print")
