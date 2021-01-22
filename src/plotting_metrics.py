import itertools
import matplotlib.pyplot as plt
import numpy as np
import sklearn


def plot_confusion_matrix(labels, predictions, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    cm = sklearn.metrics.confusion_matrix(labels, predictions, labels=range(len(class_names)))
    num_classes = len(class_names)
    figure = plt.figure(figsize=(num_classes, num_classes))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure
