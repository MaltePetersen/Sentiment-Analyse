import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns


def plot_history_metrics(history, metrics):
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_'+metric])
        plt.show()


def get_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label="auc="+str(auc), lw=2)
    plt.plot([0, 1], [0, 1], color="orange", lw=2, linestyle="--")
    plt.legend(loc=4)
    plt.show()

    y_pred[y_pred >= 0.85] = 1
    y_pred[y_pred < 0.85] = 0
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='.4g', cmap='Blues')
