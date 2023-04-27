import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histo(data, filename, visible_bins=20):
    '''
    plot a histogram of cube distributions.
    '''
    plt.figure(figsize=(10,5))
    nr_of_bins = max(data)
    plot = sns.histplot(data, bins=nr_of_bins)
    x_ticks = [i for i in range(2, nr_of_bins+1, max(nr_of_bins//visible_bins, 1))]
    plt.xticks(x_ticks)
    
    plt.savefig(filename)
    plt.show()
