import sys
import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == '__main__':
    training_log = pd.read_csv(sys.argv[1])

    categories = ("loss",)
    try:
        n_running_average = int(sys.argv[3])
    except IndexError:
        n_running_average = 10

    seaborn.set_palette("muted")
    seaborn.set_style("whitegrid")
    fig, subplots = plt.subplots(1 + len(categories), 1, sharex=True, figsize=(8, 12))
    subplots[0].plot(training_log.index, training_log['lr'])
    subplots[0].set_title('Learning Rate')
    subplots[0].set_yscale('log')

    for i, cat in enumerate(categories):
        subplots[i + 1].plot(training_log.index[n_running_average-1:],
                             running_mean(np.asarray(training_log[cat]), n_running_average),
                             label='Training')
        if 'val_' + cat in training_log.columns:
            subplots[i + 1].plot(training_log.index[n_running_average-1:],
                                 running_mean(np.asarray(training_log['val_' + cat]), n_running_average),
                                 label='Validation')
        subplots[i + 1].set_title(cat.capitalize())
        subplots[i + 1].legend()
        subplots[i + 1].set_ylim(1, 1.5)

    fig.savefig(sys.argv[2], tight_layout=True)
