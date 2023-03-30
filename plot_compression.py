import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histo(data, filename, bins=50):
    '''
    plot a histogram of cube distributions.
    '''
    plt.figure(figsize=(10,5))
    plot = sns.histplot(data, bins=bins)
    plt.xticks([i for i in range(1, 51, 1)])
    
    plt.savefig(filename)
    plt.show()
