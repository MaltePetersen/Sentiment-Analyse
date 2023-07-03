import matplotlib.pyplot as plt

def plot_history_metrics(history, metrics):
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_'+metric])
        plt.show()