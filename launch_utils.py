""" Utility funcs, including configs-related stuff and the funcs about launching scripts."""

import xml.etree.ElementTree as ET
import subprocess

import helper_funcs

name4logs = "launch_utils"


def read_configs():
    """Reads the config files and returns a dic that looks like this: {"config key": "config value", ... }. """
    res = dict()

    tree = ET.parse('config/settings.xml')
    root_settings = tree.getroot()

    data_source = ""
    for temp in root_settings.findall("data_source"):
        data_source = str(temp.get("value"))
    log_verbosity = ""
    for temp in root_settings.findall("log_verbosity"):
        log_verbosity = str(temp.get("value"))

    res["data_source"] = data_source
    res["log_verbosity"] = log_verbosity

    tree = ET.parse('config/data_channels.xml')
    root_data_channels = tree.getroot()

    data_channels = []
    for n in root_data_channels.findall("channel"):
        name = n.get("name")
        data_channels.append(name)

    data_channels_sorted = sorted(data_channels)

    res["data_channels"] = data_channels_sorted

    return res


def conda_command(file_name_for_conda, conda_env_name):
    """ Returns a string that can be used as a subprocess command.

    The output looks like this:
    conda run -n conda_env_name python3 "full/path/to/python/script.py" & disown
    """
    res = "conda run -n " + conda_env_name + " python3 "
    res += '"' + helper_funcs.get_full_path(file_name_for_conda) + '"'
    res += ' & disown'
    return res


def run(script_filename, conda_env_name):
    """ Launches a python script in the shell in the conda environment.
        Args:
            script_filename (str): e.g. "gui.py"
            conda_env_name (str): e.g. "kitnet_env"
    """
    with open("state_controls/scripts_to_run.txt", "a") as scripts_f:
        scripts_f.write(script_filename + "\n")

    # TODO: # check if such env exists beforehand:
    # https://stackoverflow.com/questions/36539623/how-do-i-find-the-name-of-the-conda-environment-in-which-my-code-is-running

    command = conda_command(script_filename, conda_env_name)
    subprocess.run(command, shell=True)


def launch_scripts():
    """ Launches all the python scripts necessary for Thio to run.

    If you want to disable Telemanom and/or KitNET, just comment them out here and restart 0launcher.
    """
    with open("state_controls/scripts_to_run.txt", "w") as scripts_f:
        scripts_f.write("")
    try:
        run('lib_telemanom_train.py', "thio_telemanom")
        run('lib_telemanom_infer.py', "thio_telemanom")
        run('lib_KitNET_train.py', "thio_kitnet")
        run('lib_KitNET_infer.py', "thio_kitnet")
        run('gui.py', "thio_kitnet")
    except Exception as e:
        helper_funcs.append_logs(str(e), name4logs, "always")


print(read_configs())
