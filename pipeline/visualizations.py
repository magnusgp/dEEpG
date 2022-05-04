from labelFunctions import solveLabelChannelRelation
from loadFunctions import TUH_data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualizeChannelEvents(events = [], timesteps = 10000):
    """
    Visualize the events of a channel in a heatmap matrix with 0 if no event,
    1 if event and 2 if the event is of elec type.
    """
    matrix = np.zeros((len(events), timesteps))
    for i, event in enumerate(events['label']):
        start = int(round(events.iloc[[i]]['t_start'],0))
        end = int(round(events.iloc[[i]]['t_end'],0))
        if event == 'elec':
            matrix[i, start:end] = 2
        elif event != 'elec':
            matrix[i, start:end] = 1

    # Plot seaborn heatmap of matrix

    sns.heatmap(matrix, cmap='Reds')
    plt.show()

    return matrix

if __name__ == "__main__":
    path = "../TUH_data_sample/131/00013103/s001_2015_09_30/00013103_s001_t000.csv"
    TUHpath = "../TUH_data_sample"
    TUH = TUH_data(path=TUHpath)
    windowssz = 50
    elecX, elecY, windowInfo = TUH.electrodeCLFPrep(tWindow=windowssz, tStep=windowssz * .25, plot=False)
    timesteps = len(elecX)

    visualizeChannelEvents(events = solveLabelChannelRelation(path), timesteps=timesteps)