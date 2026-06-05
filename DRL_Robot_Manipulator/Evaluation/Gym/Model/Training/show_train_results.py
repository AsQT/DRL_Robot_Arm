# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../../../' + 'src' not in sys.path:
    sys.path.append('../../../../' + 'src')
# OS (Operating system interfaces)
import os
# Pandas (Data analysis and manipulation) [pip3 install pandas]
import pandas as pd
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Integrate a system of ordinary differential equations (ODE) [pip3 install scipy]
import scipy
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
try:
    import scienceplots
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False
# Custom Lib.:
#   Robotics Library for Everyone (RoLE)
#       ../RoLE/Parameters/Robot
import RoLE.Parameters.Robot as Parameters

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str
# The name of the environment mode.
#   'Default':
#       The mode called "Default" demonstrates an environment without a collision object.
#   'Collision-Free':
#       The mode called "Collision-Free" demonstrates an environment with a collision object.
CONST_ENV_MODE = 'Default'
# The name of the reinforcement learning algorithm.
#   Deep Deterministic Policy Gradient (DDPG)
#       CONST_ALGORITHM = 'DDPG' or 'DDPG_HER'
#   Soft Actor Critic (SAC)
#       CONST_ALGORITHM = 'SAC' or 'SAC_HER'
#   Twin Delayed DDPG (TD3)
#       CONST_ALGORITHM = 'TD3' or 'TD3_HER'
CONST_ALGORITHM = 'DDPG'
# The selected metric to be displayed in the graph (plot).
#   CONST_METRIC = 'rollout/success_rate', 'rollout/ep_rew_mean', 'rollout/ep_len_mean',
#                  'train/actor_loss' or 'train/critic_loss'
CONST_METRIC = 'rollout/success_rate'
# Display names for chart labels.
CONST_ROBOT_NAME = 'YASKAWA GP7'
CONST_ENV_NAME   = 'Default'
# Locate the path to the project folder (derive from script file, not cwd).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = _SCRIPT_DIR
while True:
    _parent = os.path.dirname(_PROJ_ROOT)
    if os.path.basename(_parent) == 'src' or os.path.exists(os.path.join(_parent, 'src')):
        _PROJ_ROOT = _parent
        break
    if _parent == _PROJ_ROOT:
        try:
            _PROJ_ROOT = os.getcwd().split('PyBullet_Industrial_Robotics_Gym')[0] + 'PyBullet_Industrial_Robotics_Gym'
        except IndexError:
            _PROJ_ROOT = os.getcwd()
        break
    _PROJ_ROOT = _parent
CONST_PROJECT_FOLDER = _PROJ_ROOT


def _setup_plotting_style():
    """
    Set up Matplotlib with SciencePlots if available, falling back gracefully.
    Never raises an exception; always produces a usable backend.
    """
    plt.rcParams['text.usetex'] = False
    if _HAS_SCIENCEPLOTS:
        try:
            plt.style.use(['science', 'no-latex'])
            return
        except Exception:
            pass
    try:
        plt.style.use('default')
    except Exception:
        pass


def _metric_ylabel(metric: str) -> str:
    """Return a human-readable Y-axis label for a given SB3 metric name."""
    labels = {
        'rollout/success_rate': 'Mean Success Rate During Training',
        'rollout/ep_rew_mean':  'Mean Training Reward per Episode',
        'rollout/ep_len_mean':  'Mean Episode Length',
        'train/actor_loss':      'Actor Loss',
        'train/critic_loss':     'Critic Loss',
    }
    return labels.get(metric, metric)


def _plot_metric(data: pd.DataFrame, metric_name: str, y_label: str,
                  chart_title: str, line_label: str,
                  env_name: str, robot_name: str) -> bool:
    """
    Plot a single metric from training data.

    Returns True if plotted successfully, False if skipped.
    """
    if metric_name not in data.columns:
        print(f'[WARNING] Metric column "{metric_name}" not found in CSV.')
        print(f'          Available columns: {list(data.columns)}')
        return False

    x_col = 'time/total_timesteps'
    plot_data = data[[x_col, metric_name]].dropna()
    if plot_data.empty:
        print(f'[WARNING] No valid data for metric "{metric_name}" (all NaN after drop).')
        return False

    ts_vals = plot_data[x_col].values
    y_vals  = plot_data[metric_name].values
    ep_vals = np.arange(1, len(ts_vals) + 1)

    stride = max(1, len(ts_vals) // 500)
    ts_thin = ts_vals[::stride]
    y_thin  = y_vals[::stride]

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(f'{chart_title} [{robot_name} | {env_name}]')

    if len(ts_vals) > 1:
        interp_ts = np.linspace(ts_vals.min(), ts_vals.max(), 500)
        f_interp  = scipy.interpolate.interp1d(ts_vals, y_vals,
                                                kind='linear', fill_value='extrapolate')
        y_smooth  = f_interp(interp_ts)
        ax.plot(interp_ts, y_smooth, '-', color='#aeaeae',
                linewidth=1.0, alpha=0.6, label="_nolegend_")

    ax.plot(ts_thin, y_thin, '.-', color='#1565C0',
            markersize=4, label=line_label)

    ax.set_xlabel('Total Number of Timesteps', fontsize=12, labelpad=8)
    ax.set_ylabel(y_label, fontsize=12, labelpad=8)
    ax.grid(which='major', linewidth=0.15, linestyle='--')
    ax.legend(fontsize=10)

    max_ts = int(np.nanmax(ts_vals))
    tick_step = 100_000
    x_start = 0
    x_end = int(np.ceil(max_ts / tick_step) * tick_step)
    x_ticks = np.arange(x_start, x_end + tick_step, tick_step)

    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{int(v):,}"))
    ax.tick_params(axis="x", labelrotation=30)

    ax.minorticks_off()
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    top_tick_idx = np.linspace(0, len(ts_vals) - 1, 6, dtype=int)
    top_tick_positions = ts_vals[top_tick_idx]
    top_tick_labels = ep_vals[top_tick_idx]

    ax_top.xaxis.set_major_locator(ticker.FixedLocator(top_tick_positions))
    ax_top.set_xticklabels([str(int(v)) for v in top_tick_labels])
    ax_top.set_xlabel("Episode", fontsize=12, labelpad=8)

    ax_top.minorticks_off()
    ax_top.xaxis.set_minor_locator(ticker.NullLocator())

    fig.text(0.5, 0.01, f'{chart_title} [{robot_name} | {env_name}]',
             ha='center', va='bottom', fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.82, bottom=0.25)

    return True


def main():
    """
    Description:
        A program to show result data from the training. The metrics, such as Mean Success Rate During Training, Mean
        Training Reward per Episode, Mean Episode Length, etc., were used to evaluate the performance of the selected
        reinforcement learning algorithm.

        The program visualizes the results in a graph (plot).

        More information about the training process can be found in the script below:
            ../PyBullet_Industrial_Robotics_Gym/Training/train_{CONST_ALGORITHM}.py
    """

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # The specified path to the run folder containing logs/.
    run_folder = os.path.join(
        CONST_PROJECT_FOLDER,
        'Data', 'Training',
        f'Environment_{CONST_ENV_MODE}',
        CONST_ALGORITHM,
        'YASKAWA_GP7',
        'run'
    )

    csv_path = os.path.join(run_folder, 'logs', 'progress.csv')

    if not os.path.isfile(csv_path):
        print(f'[ERROR] CSV file not found: {csv_path}')
        exit(1)

    data = pd.read_csv(csv_path)

    x_col = 'time/total_timesteps'
    if x_col not in data.columns:
        print(f'[ERROR] Required column "{x_col}" not found in CSV.')
        print(f'        Available columns: {list(data.columns)}')
        exit(1)

    _setup_plotting_style()

    metrics = [
        (CONST_METRIC, _metric_ylabel(CONST_METRIC)),
        ('rollout/ep_rew_mean', _metric_ylabel('rollout/ep_rew_mean')),
    ]

    for metric_name, y_label in metrics:
        _plot_metric(data, metric_name, y_label, y_label, CONST_ALGORITHM,
                     CONST_ENV_NAME, CONST_ROBOT_NAME)

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
