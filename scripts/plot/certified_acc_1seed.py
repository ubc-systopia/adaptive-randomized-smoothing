import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from typing import *
import pandas as pd

"""
# command to generate plots in the paper:
python scripts/plot/certified_acc_1seed.py --logpath "logs/XXXX/certification_log_50000.txt" --outdir . 
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
parser.add_argument('--logpath', type=str, help='path for the certified acc log txt file, e.g. xxx/certification_log_50000.txt')
parser.add_argument('--outdir', type=str, help='dir path for saving the fig')
args = parser.parse_args()

def at_radius(df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["linf_radius"] >= radius)).mean()

x_label = r"$L_{\infty}$ radius"
max_radius=10/255
radius_step=0.0002
figtext = "None"
radii = np.arange(0, max_radius + radius_step, radius_step)

df1 = pd.read_csv(args.logpath, delimiter=" \t ")
line1 = np.array([at_radius(df1, radius) for radius in radii])

# start the figure
plt.figure()
plt.plot(radii, line1, label="ARS (ours)", color='green')

tick_frequency = 0.01
plt.ylim((0, 1))
plt.xlim((0, max_radius))
plt.tick_params()
plt.xlabel(x_label, labelpad=20, fontsize=BIGGER_SIZE)
plt.ylabel("Certified Accuracy", labelpad=20, fontsize=BIGGER_SIZE)
if figtext != "None":
    plt.figtext(0.05, 0.05, figtext)
plt.xticks(np.arange(0, max_radius+0.001, tick_frequency))
plt.legend(loc='upper right',fontsize="15")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir,"acc_1seed.png"), dpi=300)
plt.close()
