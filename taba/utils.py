import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Defining helper functions
def plot_data(errors, max, min):
    """
    Plot the erros data through extensive training
    """
    xlabel = "Hours in Future"
    tested_sequence = range(1, 13)
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Errors per Hours in Future", fontsize=20)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.setp(axs, xticks=tested_sequence, xticklabels=tested_sequence, xlabel=xlabel)

    # Plotting training loss
    print(f"errors: {len(errors[0, :])}, x: {len(tested_sequence)} ")
    sns.lineplot(
        x=tested_sequence,
        y=errors[0, :],
        ax=axs[0, 0],
        label="Mean Squared Error",
        color="red",
    )
    # axs[0,0].ticklabel_format(axis='both', style='sci', scilimits=(4,1))
    axs[0, 0].grid()
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_ylabel("MSE")

    # Plotting training MAE (denormalized)
    mae_denormalized = denormalize(errors[1, :], max, min)
    sns.lineplot(
        x=tested_sequence,
        y=mae_denormalized,
        ax=axs[0, 1],
        label="Error in Dollars",
        color="green",
    )
    axs[0, 1].set_title("MAE (denormalized)")
    axs[0, 1].grid()
    axs[0, 1].set_ylabel("US$")

    # Plotting training MAPE
    sns.lineplot(x=tested_sequence, y=errors[2, :], ax=axs[1, 0], color="blue")
    axs[1, 0].set_title("Mean Absolute Percentage Error")
    axs[1, 0].grid()
    axs[1, 0].set_ylabel("Percentage")

    # Plotting training R2 score
    sns.lineplot(x=tested_sequence, y=errors[3, :], ax=axs[1, 1], color="orange")
    axs[1, 1].set_title("R2 Score")
    axs[1, 1].grid()
    axs[1, 1].set_ylabel("R2 Score")

    plt.legend()
    plt.show()


def normalize(dataset, column_wise=False):
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
            maximum = data[:, i].max() / factor
            minimum = data[:, i].min() / factor
            data[:, i] -= minimum
            data[:, i] /= maximum - minimum
            scaler[i] = [maximum, minimum]
    else:
        maximum = data.max() / factor
        minimum = data.min() / factor
        data -= minimum
        data /= maximum - minimum
        scaler = np.array((maximum, minimum))

    return data, scaler


def denormalize(data, maximum, minimum):
    """
    Denormalize the data using the maximum and minimum values.
    Returns: Numpy array of denormalized data.
    """
    factor = 1.0
    data *= maximum * factor - minimum * factor
    data += minimum
    return data
