from sklearn.externals import joblib
from matplotlib import pyplot as plt

import argparse

def configure_axis(ax):
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.grid()

def plot_max(ax, gen, fit_max, color, label):
    ax.plot(gen, fit_max, color, label=label)
    ax.legend()

def plot_std(ax, gen, fit_std, color, label):
    ax.plot(gen, fit_std, color, label=label)
    ax.legend(loc='upper right')

def plot_avg(ax, gen, fit_avg, color, label):
    ax.plot(gen, fit_avg, color, label=label)
    ax.legend()

ap = argparse.ArgumentParser()

ap.add_argument('-t', '--tests', required=True, nargs=3)

args = vars(ap.parse_args())

colors = ['b-', 'g-', 'r-']
labels = ['Baboon', 'Jet', 'Pepper']
gen = list()
fit_max = list()
fit_std = list()
fit_avg = list()

for t in args['tests']:
    pkl = joblib.load(t)
    lb = pkl['logbook']

    gen.append(lb.select('gen'))
    fit_max.append(lb.select('max'))
    fit_std.append(lb.select('std'))
    fit_avg.append(lb.select('avg'))

(figure, axes) = plt.subplots(1, 3)

axes[0].set_title('Max individuals')
axes[1].set_title('Avg individuals')
axes[2].set_title('Std individuals')

for ax in axes:
    configure_axis(ax)

for (c, l, g, fmax, fstd, favg) in zip(colors, labels, gen, fit_max, fit_std, fit_avg):
    plot_max(axes[0], g, fmax, c, l)
    plot_avg(axes[1], g, favg, c, l)
    plot_std(axes[2], g, fstd, c, l)

plt.show()
