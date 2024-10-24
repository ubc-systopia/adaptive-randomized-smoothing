import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import *
import pandas as pd

"""
need to make sure you move the 3 seed logs to one folder as structured:
ars
├── seed1.txt
├── seed2.txt
└── seed3.txt
python3 scripts/plot/certified_acc_3seed.py --outdir . --ars <ars path> --static <static path> --cohen <cohen path>
"""
# matplotlibrc params to set for better, bigger, clear plots
SMALLER_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)   # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


parser = argparse.ArgumentParser(description='Plot some plots')
parser.add_argument('--ars', type=str, help='dir path for ars log for 3 seeds')
parser.add_argument('--static', type=str, help='dir path for static log for 3 seeds')
parser.add_argument('--cohen', type=str, help='dir path for cohen log for 3 seeds')
parser.add_argument('--outdir', type=str, help='dir path for file name')
args = parser.parse_args()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


def at_radius(df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["linf_radius"] >= radius)).mean()
    
x_label = r"$L_{\infty}$ radius"
max_radius=10/255
radius_step=0.0002
figtext = "None"
radii = np.arange(0, max_radius + radius_step, radius_step)
seeds = ["seed1.txt", "seed2.txt", "seed3.txt"]

# for ars logs
arr = []
ars_acc = {}
ars_path = args.ars
for seed in seeds :
    path = os.path.join(ars_path, seed)
    df = pd.read_csv(path, delimiter=" \t ", engine='python')
    line = np.array([at_radius(df, radius) for radius in radii])
    arr.append(line)
stacked_arr = np.vstack(arr)
ars_acc["mean"] = stacked_arr.mean(axis=0)
ars_acc["standard_test_acc"] = ars_acc["mean"][0]      
ars_acc["std_dev"] = stacked_arr.std(axis=0)
ars_acc["lower"] = ars_acc["mean"] - ars_acc["std_dev"]
ars_acc["upper"] = ars_acc["mean"] + ars_acc["std_dev"] 

# for static logs
arr = []
static_acc = {}
static_path = args.static
for seed in seeds :
    path = os.path.join(static_path, seed)
    df = pd.read_csv(path, delimiter=" \t ", engine='python')
    line = np.array([at_radius(df, radius) for radius in radii])
    arr.append(line)
stacked_arr = np.vstack(arr)
static_acc["mean"] = stacked_arr.mean(axis=0)
static_acc["standard_test_acc"] = static_acc["mean"][0]      
static_acc["std_dev"] = stacked_arr.std(axis=0)
static_acc["lower"] = static_acc["mean"] - static_acc["std_dev"]
static_acc["upper"] = static_acc["mean"] + static_acc["std_dev"] 

# for cohen logs
arr = []
cohen_acc = {}
cohen_path = args.cohen
for seed in seeds :
    path = os.path.join(cohen_path, seed)
    df = pd.read_csv(path, delimiter="\t")
    line = np.array([at_radius(df, radius) for radius in radii])
    arr.append(line)
stacked_arr = np.vstack(arr)
cohen_acc["mean"] = stacked_arr.mean(axis=0)
cohen_acc["standard_test_acc"] = cohen_acc["mean"][0]      
cohen_acc["std_dev"] = stacked_arr.std(axis=0)
cohen_acc["lower"] = cohen_acc["mean"] - cohen_acc["std_dev"]
cohen_acc["upper"] = cohen_acc["mean"] + cohen_acc["std_dev"] 


# start the figure
plt.figure()
plt.plot(radii, cohen_acc["mean"], label="Cohen at al.", color="blue")
plt.fill_between(radii, cohen_acc["lower"], cohen_acc["upper"], color='blue', alpha=0.4, linewidth=0)
plt.plot(radii, static_acc["mean"], label="Static", color="orange")
plt.fill_between(radii, static_acc["lower"], static_acc["upper"], color="orange", alpha=0.4, linewidth=0)
plt.plot(radii, ars_acc["mean"], label="ARS (Ours)", color="green")
plt.fill_between(radii, ars_acc["lower"], ars_acc["upper"], color="green", alpha=0.4, linewidth=0)

tick_frequency = 0.02
plt.ylim((0, 1))
plt.xlim((0, max_radius))
plt.tick_params()
plt.xlabel(x_label, labelpad=20, fontsize=BIGGER_SIZE)
plt.ylabel("Certified Accuracy", labelpad=20, fontsize=BIGGER_SIZE)
if figtext != "None":
    plt.figtext(0.05, 0.05, figtext)
plt.xticks(np.arange(0, max_radius+0.001, tick_frequency))
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right',fontsize="15")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir,"acc_3seeds.png"), dpi=300)
plt.close()
