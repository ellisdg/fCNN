import sys
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    training_log = pd.read_csv(sys.argv[1])

    categories = ("loss",)

    fig, subplots = plt.subplots(1 + len(categories), 1, sharex=True, figsize=(8, 12))
    subplots[0].plot(training_log.index, training_log['lr'])
    subplots[0].set_title('Learning Rate')

    for i, cat in enumerate(categories):
        subplots[i + 1].plot(training_log.index, training_log[cat],
                             label='Training')
        subplots[i + 1].plot(training_log.index, training_log['val_' + cat],
                             label='Validation')
        subplots[i + 1].set_title(cat.capitalize())
        subplots[i + 1].legend()

    fig.savefig(sys.argv[2], tight_layout=True)
