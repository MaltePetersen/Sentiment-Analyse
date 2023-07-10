import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns


def plot_history_metrics(history, metrics):
    """
    Plot the history of the metrics of the model

    Args:
        history: keras history object.
        metrics: list of strings, the metrics to plot.
    """

    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_'+metric])
        plt.show()


def get_classification_report(model, X_test, y_test, y_pred_bound=0.85):     
    """
    Plot the ROC curve and the confusion matrix of the model

    Args:
        model: keras model object.
        X_test: numpy array, the test data.
        y_test: numpy array, the test labels.
        y_pred_bound: float, the threshold for the predicted probabilities.
    """

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label="auc="+str(auc), lw=2)
    plt.plot([0, 1], [0, 1], color="orange", lw=2, linestyle="--")
    plt.legend(loc=4)
    plt.show()

    y_pred[y_pred >= y_pred_bound] = 1
    y_pred[y_pred < y_pred_bound] = 0
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='.4g', cmap='Blues')


def plot_ngrams_counts(ngrams, plot_title):
    """
    Function to plot the most common n-grams.
    
    Args:
        ngrams: List of tuples, the most common n-grams and their counts. E.g. [(('in', 'the'), 100), (('of', 'the'), 90), ...] for Bigrams.
        plot_title: String, the title of the plot.
    """

    # Separate the n-grams and their counts for plotting
    ngrams_list = [ngram[0] for ngram in ngrams]
    counts = [ngram[1] for ngram in ngrams]

    # Format the n-grams for plotting, the ngrams are in format
    ngrams_list = [' '.join(ngram) for ngram in ngrams_list]

    # Get the n of the n-grams
    n = len(ngrams_list[0].split())

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=ngrams_list)
    plt.xlabel('Count')
    plt.ylabel(f'{n}-gram')
    plt.title(plot_title)
    plt.show()


def plot_words_counts(words, plot_title):
    """
    Function to plot the most common words.
    
    Args:
        words: List of tuples, the most common words and their counts. E.g. [('the', 100), ('of', 90), ...]
        plot_title: String, the title of the plot.
    """

    # Separate the words and their counts for plotting
    words_list = [word[0] for word in words]
    counts = [word[1] for word in words]

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words_list)
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.title(plot_title)
    plt.show()