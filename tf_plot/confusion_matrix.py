import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


class ConfusionMatrixViz:
    def _create(self, est, X, y):
        y_pred = est.predict(X)
        c_matrix = confusion_matrix(y, y_pred)

        c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1,
                                                           keepdims=True)
        classes = unique_labels(y)

        self.c_matrix_ = c_matrix
        self.classes_ = classes
        return self

    def plot(self, ax=None, cmap='viridis', include_values=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        cmap = plt.get_cmap(cmap)
        c_matrix_ = self.c_matrix_
        classes_ = self.classes_

        im = ax.imshow(c_matrix_, interpolation='nearest', cmap=cmap)

        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(c_matrix_.shape[1]),
               yticks=np.arange(c_matrix_.shape[0]),
               xticklabels=classes_,
               yticklabels=classes_,
               ylabel='True label',
               xlabel='Predicted label')

        if include_values:
            thresh = c_matrix_.max() / 2.
            for i in range(c_matrix_.shape[0]):
                for j in range(c_matrix_.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(c_matrix_[i, j], '.2f'),
                        ha="center",
                        va="center",
                        color="white" if c_matrix_[i, j] < thresh else "black")
        self.ax_ = ax
        self.figure_ = ax.figure
        self.im_ = im
        return self

    @classmethod
    def _plot(cls, est, X, y, cmap='viridis', ax=None, include_values=False):
        viz = cls()._create(est, X, y)
        viz.plot(ax=ax, cmap=cmap, include_values=include_values)
        return viz


plot_confusion_matrix = ConfusionMatrixViz._plot
