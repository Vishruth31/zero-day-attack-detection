import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def dataframe_drop_correlated_columns(df, threshold=0.95):
    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=to_drop)

    return df, to_drop


def plot_probability_density(array, output_file, cutoffvalue=2):
    array[array > cutoffvalue] = cutoffvalue

    plt.clf()

    sns.histplot(array, kde=True, stat="density")

    plt.xlabel("MSE")
    plt.ylabel("Density")
    plt.title("MSE Distribution")

    plt.savefig(output_file)


def plot_model_history(hist, output_file):

    plt.clf()

    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    plt.title("Training vs Validation Loss")

    plt.savefig(output_file)