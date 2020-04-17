""" The GUI. Includes plots that are updated in real-time. Runs in parallel with training and inference scripts.

 The real-time plotting functionality is based on this code (K.Mulier, 2019): https://stackoverflow.com/a/38486349
 Our changes include:
 - it now works with an arbitrary number of sub-plots
 - it works even if some data is missing
 - the limits for y-axis are calculated automatically
 - other adaptions for our purposes
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import matplotlib

from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading

from helper_funcs import append_logs, read_prediction_for_aggregation
from helper_funcs import read_latest_points, price_dicts_list, latest_prices_for_this_crypto
from helper_funcs import get_max_anomaly_from_latest, get_full_path, exit7, synthetic_data7, \
    get_scaling_factors, get_latest_anomaly_scores
from launch_utils import read_configs
import self_diagnostics

matplotlib.use("Qt5Agg")

use_synthetic_data7 = synthetic_data7()

name4logs = "realtime_graph"

cryptos = read_configs()["data_channels"]

additional_plots = [
    "KitNET risk score",
    "Telemanom risk score"
]

ch_names = cryptos + additional_plots

if use_synthetic_data7:
    filename = "dataset/syntheticData.txt"
    anomaly_file_postfix = "_synthetic"
else:
    filename = "dataset/fetchedData.txt"
    anomaly_file_postfix = "_fetched"

sleep_time = 1.0  # in seconds


def plot_limits(historical_values):
    """Returns a tuple with the lowest and highest value for the y-axis of the plot, to make the plot better looking.

    Args:
        historical_values (list of floats): using the past values, we can guess how to better draw the plot.
            e.g. if it's [5, 3, 1, 2, 1000], limit the y-axis to between 1 and 5, and ignore the outlier 1000
    """
    y_limits = (0.000001, 100000.0)

    if len(historical_values) > 100:  # use only the most recent data, as it's more representative of the future data
        past_values_list = historical_values[:-100]
    else:
        past_values_list = historical_values

    if len(past_values_list) > 0:

        min_val, max_val = get_scaling_factors(past_values_list)

        if min_val != 0 and max_val != 0:  # if any of them is zero, the log scaling will not work
            y_limits = (min_val, max_val)
    return y_limits


# limits for the plots of the raw input data
y_lims = []
lines = read_latest_points(filename, 200)
latest_price_dicts_list = price_dicts_list(lines)
latest_dic, timestamp = latest_price_dicts_list[-1]
for i in range(len(cryptos)):
    key = cryptos[i]
    # TODO: don't read the file several times
    latest_prices = latest_prices_for_this_crypto(latest_price_dicts_list, key)
    limits = plot_limits(latest_prices)
    y_lims.append(limits)

# limits for the plot of kitnet
ts_and_score_list = get_latest_anomaly_scores("risk_scores/kitnet_anomaly" + anomaly_file_postfix + ".txt", 200)
scores = []
for gma in range(len(ts_and_score_list)):
    ts, score = ts_and_score_list[gma]
    scores.append(score)
minv, maxv = plot_limits(scores)
y_lims.append((minv, maxv * 10))  # *10 to make the anomaly spikes look prettier

# TODO: remove code duplication
# limits for the plot of telemanom
ts_and_score_list = get_latest_anomaly_scores("risk_scores/telemanom_anomaly" + anomaly_file_postfix + ".txt", 200)
scores = []
for gma in range(len(ts_and_score_list)):
    ts, score = ts_and_score_list[gma]
    scores.append(score)
minv, maxv = plot_limits(scores)
y_lims.append((minv, maxv * 10))  # *10 to make the anomaly spikes look prettier

channels_count = len(cryptos)
plots_count = len(ch_names)


def anomaly_btn_action():
    with open(get_full_path("state_controls/summonAnomaly.txt"), "w") as f:
        f.write("3")
    return


def exit_btn_action():
    with open(get_full_path("state_controls/exit7.txt"), "w") as f:
        f.write("1")
    exit7()
    return


def get_status_text():
    dead_scripts_list = self_diagnostics.get_dead_scripts()
    if len(dead_scripts_list) > 0:
        dead_scripts_str = "CAUTION: some scripts are not running: " + str(dead_scripts_list) + " . "
    else:
        dead_scripts_str = "All processes are running. "

    res = dead_scripts_str + "\n" + self_diagnostics.get_wenn_models_were_updated_string()

    return res


class CustomMainWindow(QMainWindow):
    """ Defines the window, buttons, and the callback func."""

    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 800)
        self.setWindowTitle("Thio")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210, 210, 235, 255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        # Place the widgets
        self.zoomBtn = QPushButton(text='Add an anomaly to imput data')
        self.exitBtn = QPushButton(text='Exit all processes')
        self.status_label = QLabel()
        self.status_label.setText("Hello World")
        self.zoomBtn.clicked.connect(anomaly_btn_action)
        self.exitBtn.clicked.connect(exit_btn_action)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0, 0))
        self.LAYOUT_A.addWidget(self.exitBtn, *(1, 0))
        self.LAYOUT_A.addWidget(self.status_label, *(2, 0))
        # Place the matplotlib figure
        self.myFig = RealtimeCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0, 1))
        # Add the callback func to ..
        data_loop = threading.Thread(name='myDataLoop', target=data_send_loop, daemon=True,
                                     args=(self.add_data_callback_func,))
        data_loop.start()
        self.show()
        return

    def add_data_callback_func(self, value):
        self.myFig.add_data(value)
        self.status_label.setText(get_status_text())
        return


class RealtimeCanvas(FigureCanvas, TimedAnimation):
    """ The class responsible for the real-time plotting."""

    def __init__(self):

        added_data_all = []
        for c in range(plots_count):
            added_data_all.append([])

        self.added_data_all = added_data_all

        print(matplotlib.__version__)
        self.xlim = 200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        self.fig = Figure(figsize=(5, 8), dpi=100)

        y_all = []
        ax_all = []
        line_all = []
        line_tail_all = []
        line_head_all = []

        for c in range(plots_count):
            y_all.append(self.n * 0.0)

            ax = self.fig.add_subplot(plots_count, 1, c + 1)

            ax.set_xlabel(str(c))
            ax.set_ylabel(ch_names[c], rotation='horizontal', labelpad=-60, ha='left')

            line = Line2D([], [], color='blue', linewidth=1)
            line_tail = Line2D([], [], color='red', linewidth=1)
            line_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')

            line_all.append(line)
            line_tail_all.append(line_tail)
            line_head_all.append(line_head)

            ax.add_line(line)
            ax.add_line(line_tail)
            ax.add_line(line_head)
            ax.set_xlim(0, self.xlim - 1)
            ax.set_ylim(y_lims[c])
            ax.set_yscale('log')

            ax_all.append(ax)

        self.y_all = y_all
        self.ax_all = ax_all
        self.line_all = line_all
        self.line_tail_all = line_tail_all
        self.line_head_all = line_head_all

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=50, blit=True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines_all = self.line_all + self.line_tail_all + self.line_head_all
        for la in lines_all:
            la.set_data([], [])
        return

    def add_data(self, value):

        prices_dic = value[0]
        if prices_dic is not None:
            for c in range(len(cryptos)):
                if cryptos[c] in prices_dic:
                    self.added_data_all[c].append(prices_dic[cryptos[c]])
                else:
                    self.added_data_all[c].append(None)
        else:
            for c in range(len(cryptos)):
                self.added_data_all[c].append(None)

        kitnet_risk = value[1]
        c = len(cryptos)
        self.added_data_all[c].append(kitnet_risk)

        telemanom_risk = value[2]
        self.added_data_all[c + 1].append(telemanom_risk)

        return

    def _step(self, *args):
        """ Extends the _step() method for the TimedAnimation class."""
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2

        while len(self.added_data_all[0]) > 0:
            for c in range(plots_count):
                self.y_all[c] = np.roll(self.y_all[c], -1)
                self.y_all[c][-1] = self.added_data_all[c][0]
                del (self.added_data_all[c][0])

                temp_ax = self.ax_all[c]

                value_str = " : " + str(round(self.y_all[c][-1], 4))

                temp_ax.set_ylabel(ch_names[c] + value_str, rotation='horizontal', labelpad=-60, ha='left')
                self.ax_all[c] = temp_ax

        artists_all = []
        for c in range(plots_count):
            self.line_all[c].set_data(self.n[0: self.n.size - margin], self.y_all[c][0: self.n.size - margin])
            self.line_tail_all[c].set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]),
                                           np.append(self.y_all[c][-10:-1 - margin], self.y_all[c][-1 - margin]))
            self.line_head_all[c].set_data(self.n[-1 - margin], self.y_all[c][-1 - margin])
            artists_all.append(self.line_all[c])
            artists_all.append(self.line_tail_all[c])
            artists_all.append(self.line_head_all[c])
            artists_all.append(self.ax_all[c])

        self._drawn_artists = artists_all
        return


class Communicate(QObject):
    """Sets up a signal slot mechanism, to send data to the GUI in a thread-safe way."""
    data_signal = pyqtSignal(list)


def data_send_loop(add_data_callback_func):
    """Regularly reads the data to plot, and emits it."""

    # Setup the signal-slot mechanism.
    source = Communicate()
    source.data_signal.connect(add_data_callback_func)

    loop_counter = 0

    if use_synthetic_data7:
        data_filename = "dataset/latest_datapoint_synthetic.txt"
    else:
        data_filename = "dataset/latest_datapoint_fetched.txt"

        # TODO: generate this dic automatically
    pause_fetching = {
        "prices_dic": False,
        "kitnet_risk": False,
        "telemanom_risk": False
    }

    while True:
        # TODO: generate this list and the dicts automatically
        list2emit = [None, None, None]
        try:
            prices_dic = dict()
            kitnet_risk = dict()
            telemanom_risk = dict()

            # to prevent flooding the log with the entries about non-existing file during the first start
            if not pause_fetching["prices_dic"]:
                prices_dic = read_prediction_for_aggregation(data_filename, "realtime_graph")
                if prices_dic is None:
                    pause_fetching["prices_dic"] = True

            if not pause_fetching["kitnet_risk"]:
                # TODO: calculate the number from configs
                kitnet_risk = get_max_anomaly_from_latest("risk_scores/kitnet_anomaly" + anomaly_file_postfix + ".txt",
                                                          10)
                if kitnet_risk is None:
                    pause_fetching["kitnet_risk"] = True

            if not pause_fetching["telemanom_risk"]:
                # TODO: calculate the number from configs
                telemanom_risk = get_max_anomaly_from_latest(
                    "risk_scores/telemanom_anomaly" + anomaly_file_postfix + ".txt", 10)
                if telemanom_risk is None:
                    pause_fetching["telemanom_risk"] = True

            list2emit = [prices_dic, kitnet_risk, telemanom_risk]

            append_logs(str(prices_dic), name4logs, "verbose")
        except Exception as e:
            append_logs(str(e), name4logs, "always")

        time.sleep(sleep_time)  # in seconds
        source.data_signal.emit(list2emit)  # <- Here you emit a signal!

        loop_counter += 1

        if loop_counter % 60 == 0:
            for channel_key, value in pause_fetching.items():
                pause_fetching[channel_key] = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())
