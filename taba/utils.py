import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Defining helper functions
def plot_data(data, title = 'Data', xlabel = 'Date', ylabel = 'Price'):
    """
    Plot the data with the given title and labels.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(data['datetime'], data['close'], label='Close Price')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def normalize(dataset, column_wise = False):
    """
    Normalize the data to the range [0, 1].
    If column_wise is True, normalize each column separately.
    Returns: Numpy array of normalized data, maximum and minimum values used for normalization.
    """
    data = dataset.to_numpy()
    factor = 1.0
    if column_wise:
        scaler = np.zeros((data.shape[1], 2))
        for i in range(data.shape[1]):
            maximum = data[:,i].max()/factor
            minimum = data[:,i].min()/factor
            data[:,i] -= minimum
            data[:,i] /= (maximum - minimum)
            scaler[i] = [maximum, minimum]
    else:
        maximum = data.max()/factor
        minimum = data.min()/factor
        data -= minimum
        data /= (maximum - minimum)
        scaler = np.array((maximum, minimum))

    return data, scaler

def denormalize(data, maximum, minimum):
    """
    Denormalize the data using the maximum and minimum values.
    Returns: Numpy array of denormalized data.
    """
    factor = 1.0
    data *= (maximum*factor - minimum*factor)
    data += minimum
    return data