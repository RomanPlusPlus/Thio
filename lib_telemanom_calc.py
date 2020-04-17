""" Almost everything related to Telemanom is done in this module.

Telemanom is framework for using LSTMs to detect anomalies in multivariate time series data, invented by [Hundman et al, 2018].

Most of the code below is a modified version of their code, released under an Apache 2.0 license.
The corresponding license text is at end of this file.
Source: https://github.com/khundman/telemanom
Paper: Hundman, Constantinou, Laporte, Colwell, Soderstrom. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. KDD '18: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data MiningJuly 2018 Pages 387–395. https://arxiv.org/abs/1802.04431
"""

from datetime import datetime as dt
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
import json
import logging
import more_itertools as mit
import numpy as np
import os
import pandas as pd
import sys
import yaml
import argparse
import traceback

import launch_utils
import dataset_preprocessing as dsp
from helper_funcs import append_logs, exit7

name4logs = "lib_telemanom_calc"

logger = logging.getLogger('telemanom')

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --------------------------------------CHANNEL------------------------

class Channel:
    def __init__(self, config, chan_id):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data
            test(arr): test data 
            scale_lower (float) = None
            scale_upper (float) = None
        """

        self.id = chan_id
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None
        self.scale_lower = None
        self.scale_upper = None
        self.bad_data = False

    def shape_data(self, arr, train=True):
        """Shape raw input streams for ingestion into LSTM. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """

        data = []
        # TODO: check for cases where arr is too short, making the range arg negative
        len_for_range = len(arr) - self.config.l_s - self.config.n_predictions
        if len_for_range > 0:
            for i in range(len_for_range):
                data.append(arr[i:i + self.config.l_s + self.config.n_predictions])
            data = np.array(data)

            assert len(data.shape) == 3

            if train:
                np.random.shuffle(data)
                self.X_train = data[:, :-self.config.n_predictions, :]
                self.y_train = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0
            else:
                self.X_test = data[:, :-self.config.n_predictions, :]
                self.y_test = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0
        else:
            msg = "Caution: len_for_range is <= 0. Usually nothing to worry about, as the input-data scaling process could produse dataframes too small for LSTM"
            append_logs(msg, name4logs, "always", "print")
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            self.bad_data = True

    def load_data(self, observations_for_inference=None, scaling_factors=None, training_datapoints=None):
        """
        Load train and test data from local.
        """
        try:
            if observations_for_inference is not None:
                append_logs("inference data is from RAM", name4logs, "always", "print")
                all_data_df = observations_for_inference

            if training_datapoints is not None:
                append_logs("training data is from RAM", name4logs, "always", "print")
                all_data_df = training_datapoints

            msg = "all_data_df.tail:\n" + str(all_data_df.tail())
            append_logs(msg, name4logs, "always", "print")

            # TODO: ckeck for the case where there is no such column
            raw_df = all_data_df[[self.id]]

            msg = "Number of datapoints for " + str(self.id) + " :" + str(len(raw_df.index))
            append_logs(msg, name4logs, "always", "print")
            msg = "scaling_factors:" + str(scaling_factors)
            append_logs(msg, name4logs, "always", "print")

            one_channel_df, scale_lower, scale_upper = dsp.normilize_single_channel_df(raw_df, scaling_factors)

            c = []
            c.extend(range(0, 24))
            c = [str(i) for i in c]
            one_channel_df = one_channel_df.assign(**dict.fromkeys(c, 0))

            channel_np = one_channel_df.to_numpy()

            if observations_for_inference is None:  # = will train, not infer
                train_np = channel_np
                self.train = train_np
                self.shape_data(self.train)
            else:  # = will infer, not train
                test_np = channel_np
                self.test = test_np
                self.shape_data(self.test, train=False)
            self.scale_lower = scale_lower
            self.scale_upper = scale_upper

        except Exception as e:
            msg = "Exception in def load_data(self): " + str(e) + " " + str(traceback.print_exc())
            append_logs(msg, name4logs, "always", "print")
            logger.critical(e)
            logger.critical(msg)


# ---------------------------- ERRORS-------------------------------


class Errors:
    def __init__(self, channel, config, run_id):
        """
        Batch processing of errors between actual and predicted values
        for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            run_id (str): Datetime referencing set of predictions in use

        Attributes:
            config (obj): see Args
            window_size (int): number of trailing batches to use in error
                calculation
            n_windows (int): number of windows in test values for channel
            i_anom (arr): indices of anomalies in channel test values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in test values
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq
            e (arr): errors in prediction (predicted - actual)
            e_s (arr): exponentially-smoothed errors in prediction
            normalized (arr): prediction errors as a percentage of the range
                of the channel values
        """

        self.config = config
        self.window_size = self.config.window_size
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []

        if not channel.bad_data:
            self.n_windows = int((channel.y_test.shape[0] -
                                  (self.config.batch_size * self.window_size))
                                 / self.config.batch_size)

            # raw prediction error
            self.e = [abs(y_h - y_t[0]) for y_h, y_t in
                      zip(channel.y_hat, channel.y_test)]

            smoothing_window = int(self.config.batch_size * self.config.window_size
                                   * self.config.smoothing_perc)
            if not len(channel.y_hat) == len(channel.y_test):
                raise ValueError('len(y_hat) != len(y_test): {}, {}'
                                 .format(len(channel.y_hat), len(channel.y_test)))

            # smoothed prediction error
            self.e_s = pd.DataFrame(self.e).ewm(span=smoothing_window) \
                .mean().values.flatten()

            # for values at beginning < sequence length, just use avg
            if not channel.id == 'C-2':  # anomaly occurs early in window
                self.e_s[:self.config.l_s] = \
                    [np.mean(self.e_s[:self.config.l_s * 2])] * self.config.l_s

            self.normalized = np.mean(self.e / np.ptp(channel.y_test))
            logger.info("normalized prediction error: {0:.2f}"
                        .format(self.normalized))

        else:
            self.n_windows = None
            self.e = None
            self.e_s = None
            self.normalized = None

    def get_raw_prediction_errors(self):
        return self.e

    def adjust_window_size(self, channel):
        """
        Decrease the historical error window size (h) if number of test
        values is limited.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        if not channel.bad_data:

            while self.n_windows < 0:
                self.window_size -= 1
                self.n_windows = int((channel.y_test.shape[0]
                                      - (self.config.batch_size * self.window_size))
                                     / self.config.batch_size)
                if self.window_size == 1 and self.n_windows < 0:
                    raise ValueError('Batch_size ({}) larger than y_test (len={}). '
                                     'Adjust in config.yaml.'
                                     .format(self.config.batch_size,
                                             channel.y_test.shape[0]))

    def merge_scores(self):
        """
        If anomalous sequences from subsequent batches are adjacent they
        will automatically be combined. This combines the scores for these
        initial adjacent sequences (scores are calculated as each batch is
        processed) where applicable.
        """

        merged_scores = []
        score_end_indices = []

        for i, score in enumerate(self.anom_scores):
            if not score['start_idx'] - 1 in score_end_indices:
                merged_scores.append(score['score'])
                score_end_indices.append(score['end_idx'])

    def process_batches(self, channel):
        """
        Top-level function for the Error class that loops through batches
        of values for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        if not channel.bad_data:

            self.adjust_window_size(channel)

            for i in range(0, self.n_windows + 1):
                prior_idx = i * self.config.batch_size
                idx = (self.config.window_size * self.config.batch_size) \
                      + (i * self.config.batch_size)
                if i == self.n_windows:
                    idx = channel.y_test.shape[0]

                window = ErrorWindow(channel, self.config, prior_idx, idx, self, i)

                window.find_epsilon()
                window.find_epsilon(inverse=True)

                window.compare_to_epsilon(self)
                window.compare_to_epsilon(self, inverse=True)

                if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                    continue

                window.prune_anoms()
                window.prune_anoms(inverse=True)

                if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                    continue

                window.i_anom = np.sort(np.unique(
                    np.append(window.i_anom, window.i_anom_inv))).astype('int')
                window.score_anomalies(prior_idx)

                # update indices to reflect true indices in full set of values
                self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
                self.anom_scores = self.anom_scores + window.anom_scores

            if len(self.i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups = [list(group) for group in
                          mit.consecutive_groups(self.i_anom)]
                self.E_seq = [(int(g[0]), int(g[-1])) for g in groups
                              if not g[0] == g[-1]]

                # additional shift is applied to indices so that they represent the
                # position in the original data array, obtained from the files,
                # and not the position on y_test (See PR #27).
                self.E_seq = [(e_seq[0] + self.config.l_s,
                               e_seq[1] + self.config.l_s) for e_seq in self.E_seq]

                self.merge_scores()


class ErrorWindow:
    def __init__(self, channel, config, start_idx, end_idx, errors, window_num):
        """
        Data and calculations for a specific window of prediction errors.
        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            start_idx (int): Starting index for window within full set of
                channel test values
            end_idx (int): Ending index for window within full set of channel
                test values
            errors (arr): Errors class object
            window_num (int): Current window number within channel test values

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """

        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000

        self.config = config
        self.anom_scores = []

        self.window_num = window_num

        self.sd_lim = 12.0
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        if not channel.bad_data:
            self.e_s = errors.e_s[start_idx:end_idx]

            self.mean_e_s = np.mean(self.e_s)
            self.sd_e_s = np.std(self.e_s)
            self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e)
                                     for e in self.e_s])

            self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
            self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

            self.y_test = channel.y_test[start_idx:end_idx]
            self.sd_values = np.std(self.y_test)

            self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])
            self.inter_range = self.perc_high - self.perc_low

            # ignore initial error values until enough history for processing
            self.num_to_ignore = self.config.l_s * 2
            # if y_test is small, ignore fewer
            if len(channel.y_test) < 2500:
                self.num_to_ignore = self.config.l_s
            if len(channel.y_test) < 1800:
                self.num_to_ignore = 0

    def find_epsilon(self, inverse=False):
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -10000000

        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon]

            i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )
            buffer = np.arange(1, self.config.error_buffer)
            i_anom = np.sort(np.concatenate((i_anom,
                                             np.array([i + buffer for i in i_anom])
                                             .flatten(),
                                             np.array([i - buffer for i in i_anom])
                                             .flatten())))
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            if len(i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups = [list(group) for group
                          in mit.consecutive_groups(i_anom)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) \
                                     / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) \
                                   / self.sd_e_s
                score = (mean_perc_decrease + sd_perc_decrease) \
                        / (len(E_seq) ** 2 + len(i_anom))

                # sanity checks / guardrails
                if score >= max_score and len(E_seq) <= 5 and \
                        len(i_anom) < (len(e_s) * 0.5):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, errors_all, inverse=False):
        """
        Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            :param errors_all: Errors class object containing list of all
            previously identified anomalies in test set
            :param inverse: a boolean
        """

        e_s = self.e_s if not inverse else self.e_s_inv
        epsilon = self.epsilon if not inverse else self.epsilon_inv

        # Check: scale of errors compared to values too small?
        if not (self.sd_e_s > (.05 * self.sd_values) or max(self.e_s)
                > (.05 * self.inter_range)) or not max(self.e_s) > 0.05:
            return

        i_anom = np.argwhere((e_s >= epsilon) &
                             (e_s > 0.05 * self.inter_range)).reshape(-1, )

        if len(i_anom) == 0:
            return
        buffer = np.arange(1, self.config.error_buffer + 1)
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        # if it is first window, ignore initial errors (need some history)
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.config.batch_size]

        i_anom = np.sort(np.unique(i_anom))

        # capture max of non-anomalous values below the threshold
        # (used in filtering process)
        batch_position = self.window_num * self.config.batch_size
        window_indices = np.arange(0, len(e_s)) + batch_position
        adj_i_anom = i_anom + batch_position
        window_indices = np.setdiff1d(window_indices,
                                      np.append(errors_all.i_anom, adj_i_anom))
        candidate_indices = np.unique(window_indices - batch_position)
        non_anom_max = np.max(np.take(e_s, candidate_indices))

        # group anomalous indices into continuous sequences
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        if inverse:
            self.i_anom_inv = i_anom
            self.E_seq_inv = E_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.E_seq = E_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse=False):
        """
        Remove anomalies that don't meet minimum separation from the next
        closest anomaly or error value

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """

        e_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        non_anom_max = self.non_anom_max if not inverse \
            else self.non_anom_max_inv

        if len(e_seq) == 0:
            return

        e_seq_max = np.array([max(e_s[e[0]:e[1] + 1]) for e in e_seq])
        e_seq_max_sorted = np.sort(e_seq_max)[::-1]
        e_seq_max_sorted = np.append(e_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([])
        for i in range(0, len(e_seq_max_sorted) - 1):
            if (e_seq_max_sorted[i] - e_seq_max_sorted[i + 1]) \
                    / e_seq_max_sorted[i] < self.config.p:
                i_to_remove = np.append(i_to_remove, np.argwhere(
                    e_seq_max == e_seq_max_sorted[i]))
            else:
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()

        if len(i_to_remove) > 0:
            e_seq = np.delete(e_seq, i_to_remove, axis=0)

        if len(e_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(e_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1] + 1)
                                          for e_seq in e_seq])

        if not inverse:
            mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx):
        """
        Calculate anomaly scores based on max distance from epsilon
        for each anomalous sequence.

        Args:
            prior_idx (int): starting index of window within full set of test
                values for channel
        """

        groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]

        for e_seq in groups:
            score_dict = {
                "start_idx": e_seq[0] + prior_idx,
                "end_idx": e_seq[-1] + prior_idx,
                "score": 0
            }

            score = max([abs(self.e_s[i] - self.epsilon)
                         / (self.mean_e_s + self.sd_e_s) for i in
                         range(e_seq[0], e_seq[-1] + 1)])
            inv_score = max([abs(self.e_s_inv[i] - self.epsilon_inv)
                             / (self.mean_e_s + self.sd_e_s) for i in
                             range(e_seq[0], e_seq[-1] + 1)])

            # the max score indicates whether anomaly was from regular
            # or inverted errors
            score_dict['score'] = max([score, inv_score])
            self.anom_scores.append(score_dict)

        # -----------------------------------HELPERS -------------------------


class Config:
    """Loads parameters from config.yaml into global object

    """

    def __init__(self, path_to_config):

        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = 'config/{}'.format(self.path_to_config)

        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup


def setup_logging():
    '''Configure logging object to track parameter settings, training, and evaluation.
    
    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    '''

    logger_obj = logging.getLogger('telemanom')
    logger_obj.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger_obj.addHandler(stdout)

    return logger_obj


# -------------------------------------MODELLING-------------------------


class Model:
    def __init__(self, config, run_id, channel, single_channel_model=None):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.scale_lower = channel.scale_lower
        self.scale_upper = channel.scale_upper
        self.single_channel_model = single_channel_model

        msg = "self.config.train: " + str(self.config.train)
        append_logs(msg, name4logs, "always", "print")
        if not self.config.train:
            try:
                self.load_from_ram()
            except Exception as e:
                msg = "Exception in class Model:" + str(e) + " " + str(traceback.print_exc())
                append_logs(msg, name4logs, "always", "print")
                self.train_new(channel)
                # self.save()
        else:
            self.train_new(channel)
            # self.save()

    def load_from_ram(self):
        self.model = self.single_channel_model.model
        append_logs("loaded model from RAM", name4logs, "always", "print")

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        if not channel.bad_data:
            cbs = [History(), EarlyStopping(monitor='val_loss',
                                            patience=self.config.patience,
                                            min_delta=self.config.min_delta,
                                            verbose=0)]

            self.model = Sequential()

            self.model.add(LSTM(
                self.config.layers[0],
                input_shape=(None, channel.X_train.shape[2]),
                return_sequences=True))
            self.model.add(Dropout(self.config.dropout))

            self.model.add(LSTM(
                self.config.layers[1],
                return_sequences=False))
            self.model.add(Dropout(self.config.dropout))

            self.model.add(Dense(
                self.config.n_predictions))
            self.model.add(Activation('linear'))

            self.model.compile(loss=self.config.loss_metric,
                               optimizer=self.config.optimizer)

            self.model.fit(channel.X_train,
                           channel.y_train,
                           batch_size=self.config.lstm_batch_size,
                           epochs=self.config.epochs,
                           validation_split=self.config.validation_split,
                           callbacks=cbs,
                           verbose=True)

    # def save(self):
    #     """
    #    Save trained model.
    #    """

        # self.model.save(os.path.join('data', self.run_id, 'models',
        #                             '{}.h5'.format(self.chan_id)))

        # self.model.save(os.path.join('pickled_models', 'telemanom',
        #                             '{}.h5'.format(self.chan_id))) 

        # with open(os.path.join("pickled_models", "telemanom", "{}_data_scale.txt".format(self.chan_id)), "w") as f:
        #    f.write(str(self.scale_lower) + "\n")                             
        #    f.write(str(self.scale_upper) + "\n")                                     

    def return_model(self):
        return self.model

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t + 1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        if not channel.bad_data:
            num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                              / self.config.batch_size)
            if num_batches < 0:
                raise ValueError('l_s ({}) too large for stream length {}.'
                                 .format(self.config.l_s, channel.y_test.shape[0]))

            # simulate data arriving in batches, predict each batch
            for i in range(0, num_batches + 1):
                prior_idx = i * self.config.batch_size
                idx = (i + 1) * self.config.batch_size

                if i + 1 == num_batches + 1:
                    # remaining values won't necessarily equal batch size
                    idx = channel.y_test.shape[0]

                X_test_batch = channel.X_test[prior_idx:idx]
                y_hat_batch = self.model.predict(X_test_batch)
                self.aggregate_predictions(y_hat_batch)

            self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))
            channel.y_hat = self.y_hat

        return channel

# --------------------------------------- DETECTOR ------------------------


class Detector:
    def __init__(self, labels_path=None, result_path='results/',
                 config_path='telemanom.yaml', input_metamodel=None, train_model7=True, observations_for_inference=None,
                 scaling_factors_for_inference_dic=None, training_datapoints=None):
        """
        Top-level class for running anomaly detection over a group of channels
        Also evaluates performance against a set of labels if provided.

        Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed
            result_path (str): directory indicating where to stick result .csv
            config_path (str): path to config.yaml

        Attributes:
            labels_path (str): see Args
            results (list of dicts): holds dicts of results for each channel
            result_df (dataframe): results converted to pandas dataframe
            chan_df (dataframe): holds all channel information from labels .csv
            result_tracker (dict): if labels provided, holds results throughout
                processing for logging
            config (obj):  Channel class object containing train/test data
                for X,y for a single channel
            y_hat (arr): predicted channel values
            id (str): datetime id for tracking different runs
            result_path (str): see Args
        """

        self.input_metamodel = input_metamodel
        self.models_dic = None
        self.scales_dic = None
        self.labels_path = labels_path
        self.results = []
        self.result_df = None
        self.raw_errors_dic = None
        self.chan_df = None
        self.observations_for_inference = observations_for_inference
        self.training_datapoints = training_datapoints
        self.scaling_factors_for_inference_dic = scaling_factors_for_inference_dic

        self.result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

        self.config = Config(config_path)
        self.y_hat = None

        self.config.train = train_model7
        self.config.predict = not train_model7

        if not self.config.predict and self.config.use_id:
            self.id = self.config.use_id
        else:
            self.id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.result_path = result_path

        if self.labels_path:
            self.chan_df = pd.read_csv(labels_path)
        else:
            chan_ids = launch_utils.read_configs()["data_channels"]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})

        logger.info("{} channels found for processing."
                    .format(len(self.chan_df)))

    def evaluate_sequences(self, errors, label_row):
        """
        Compare identified anomalous sequences with labeled anomalous sequences.

        Args:
            errors (obj): Errors class object containing detected anomaly
                sequences for a channel
            label_row (pandas Series): Contains labels and true anomaly details
                for a channel

        Returns:
            result_row (dict): anomaly detection accuracy and results
        """

        result_row = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'fp_sequences': [],
            'tp_sequences': [],
            'num_true_anoms': 0
        }

        matched_true_seqs = []

        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        result_row['scores'] = errors.anom_scores

        if len(errors.E_seq) == 0:
            result_row['false_negatives'] = result_row['num_true_anoms']

        else:
            true_indices_grouped = [list(range(e[0], e[1] + 1)) for e in label_row['anomaly_sequences']]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])

            for e_seq in errors.E_seq:
                i_anom_predicted = set(range(e_seq[0], e_seq[1] + 1))

                matched_indices = list(i_anom_predicted & true_indices_flat)
                valid = True if len(matched_indices) > 0 else False

                if valid:

                    result_row['tp_sequences'].append(e_seq)

                    true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] += 1

                else:
                    result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                    result_row['false_positives'] += 1

            result_row["false_negatives"] = len(np.delete(label_row['anomaly_sequences'],
                                                          matched_true_seqs, axis=0))

        logger.info('Channel Stats: TP: {}  FP: {}  FN: {}'.format(result_row['true_positives'],
                                                                   result_row['false_positives'],
                                                                   result_row['false_negatives']))

        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row

    def log_final_stats(self):
        """
        Log final stats at end of experiment.
        """

        if self.labels_path:

            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('True Positives: {}'
                        .format(self.result_tracker['true_positives']))
            logger.info('False Positives: {}'
                        .format(self.result_tracker['false_positives']))
            logger.info('False Negatives: {}\n'
                        .format(self.result_tracker['false_negatives']))
            try:
                logger.info('Precision: {0:.2f}'
                            .format(float(self.result_tracker['true_positives'])
                                    / float(self.result_tracker['true_positives']
                                            + self.result_tracker['false_positives'])))
                logger.info('Recall: {0:.2f}'
                            .format(float(self.result_tracker['true_positives'])
                                    / float(self.result_tracker['true_positives']
                                            + self.result_tracker['false_negatives'])))
            except ZeroDivisionError as e:
                msg = "Precision: NaN, Recall: NaN. " + str(e) + " " + str(traceback.print_exc())
                append_logs(msg, name4logs, "always", "print")

        else:
            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('Total channel sets evaluated: {}'
                        .format(len(self.result_df)))
            logger.info('Total anomalies found: {}'
                        .format(self.result_df['n_predicted_anoms'].sum()))
            logger.info('Avg normalized prediction error: {}'
                        .format(self.result_df['normalized_pred_error'].mean()))
            logger.info('Total number of values evaluated: {}'
                        .format(self.result_df['num_test_values'].sum()))

    def run(self):
        """
        Initiate processing for all channels.
        """

        try:
            models_dic = dict()
            raw_errors_dic = dict()
            scales_dic = dict()
            for i, row in self.chan_df.iterrows():
                exit7()

                input_single_channel_model = None

                if self.input_metamodel is not None:  # there is a model to use
                    if row.chan_id in self.input_metamodel:
                        input_single_channel_model = self.input_metamodel[row.chan_id]
                    else:
                        msg = "fail: no such model in metamodel: " + str(row.chan_id)
                        append_logs(msg, name4logs, "always", "print")
                else:
                    append_logs("self.input_metamodel is None", name4logs, "always", "print")

                channel = Channel(self.config, row.chan_id)

                if self.scaling_factors_for_inference_dic is not None:
                    # TODO: checks for cases where key is absent from the dict
                    scaling_pair_for_channel = self.scaling_factors_for_inference_dic[row.chan_id]
                else:
                    scaling_pair_for_channel = None

                channel.load_data(self.observations_for_inference, scaling_pair_for_channel, self.training_datapoints)
                scales_dic[row.chan_id] = (channel.scale_lower, channel.scale_upper)

                model = Model(self.config, self.id, channel, input_single_channel_model)
                if self.config.predict:
                    channel = model.batch_predict(channel)
                else:
                    model.train_new(channel)
                    models_dic[row.chan_id] = model
                    append_logs("Trained a channel model for " + row.chan_id, name4logs, "normal")

                if self.config.predict:
                    errors = Errors(channel, self.config, self.id)
                    errors.process_batches(channel)

                    raw_errors_dic[row.chan_id] = errors.e

                    if channel.X_train is not None:
                        num_train_values = len(channel.X_train)
                    else:
                        num_train_values = 0

                    if channel.X_test is not None:
                        num_test_values = len(channel.X_test)
                    else:
                        num_test_values = 0

                    result_row = {
                        'run_id': self.id,
                        'chan_id': row.chan_id,
                        'num_train_values': num_train_values,
                        'num_test_values': num_test_values,
                        'n_predicted_anoms': len(errors.E_seq),
                        'normalized_pred_error': errors.normalized,
                        'anom_scores': errors.anom_scores
                    }

                    if self.labels_path:
                        result_row = {**result_row,
                                      **self.evaluate_sequences(errors, row)}
                        result_row['spacecraft'] = row['spacecraft']
                        result_row['anomaly_sequences'] = row['anomaly_sequences']
                        result_row['class'] = row['class']
                        self.results.append(result_row)

                        logger.info('Total true positives: {}'
                                    .format(self.result_tracker['true_positives']))
                        logger.info('Total false positives: {}'
                                    .format(self.result_tracker['false_positives']))
                        logger.info('Total false negatives: {}\n'
                                    .format(self.result_tracker['false_negatives']))

                    else:
                        result_row['anomaly_sequences'] = errors.E_seq
                        self.results.append(result_row)

                        logger.info('{} anomalies found'
                                    .format(result_row['n_predicted_anoms']))
                        logger.info('anomaly sequences start/end indices: {}'
                                    .format(result_row['anomaly_sequences']))
                        logger.info('number of test values: {}'
                                    .format(result_row['num_test_values']))
                        logger.info('anomaly scores: {}\n'
                                    .format(result_row['anom_scores']))

                    self.result_df = pd.DataFrame(self.results)
                    self.log_final_stats()

            self.models_dic = models_dic
            self.raw_errors_dic = raw_errors_dic
            self.scales_dic = scales_dic

        except Exception as e:
            msg = "Exception in run: " + str(e) + " " + str(traceback.print_exc())
            append_logs(msg, name4logs, "always", "print")


def get_model(datapoints):
    """ Returns a new Telemanom model trained on the provided data, and a dict with data scaling factors.

    Args:
        datapoints (a Pandas dataFrame): the data to train upon. Columns correspond to channels.
    """
    arg_parser = argparse.ArgumentParser(description='Parse path to anomaly labels if provided.')
    arg_parser.add_argument('-l', '--labels_path', default=None, required=False)
    args = arg_parser.parse_args()

    detector = Detector(labels_path=args.labels_path, train_model7=True, training_datapoints=datapoints)
    detector.run()
    return detector.models_dic, detector.scales_dic, True


def ask_model(lmodel, datapoints, scaling):
    """ Returns the anomaly score (float), infered from the provided models, data for inference, and scaling factors.

    Args:
        lmodel (a dict of Telemanom models): e.g. {"bitcoin":<model_for_bitcoin>}
        datapoints (a Pandas dataFrame): the data to calculate anomaly scores from. Columns correspond to channels.
        scaling (a pair of floats): to ensure that the data for inference is scaled exactly like the data used
        to train the model.
    """
    try:

        arg_parser = argparse.ArgumentParser(description='Parse path to anomaly labels if provided.')
        arg_parser.add_argument('-l', '--labels_path', default=None, required=False)
        args = arg_parser.parse_args()

        detector = Detector(labels_path=args.labels_path, input_metamodel=lmodel, train_model7=False,
                            observations_for_inference=datapoints, scaling_factors_for_inference_dic=scaling)
        detector.run()

        errors_dic = detector.raw_errors_dic

        united_list = []
        for key, value in errors_dic.items():
            united_list = united_list + value
        rmse_score = 2 * max(united_list)

    except Exception as e:
        rmse_score = 0
        msg = "Exception in ask_model: datapoints: " + str(datapoints) + " . Exception: " + str(e) + " " + str(
            traceback.print_exc())
        append_logs(msg, name4logs, "always")
    return rmse_score

# --------------------------------------------------

# Most of the code above is a modified version of the code by [Hundman et al, 2018], which was released under the
# following license:

# Copyright Assertion

# Copyright (c) 2018, California Institute of Technology ("Caltech").  U.S. Government sponsorship acknowledged.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:

# •	Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# •	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# •	Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its
# contributors may be used to endorse or promote products derived from this software without specific prior written
# permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
