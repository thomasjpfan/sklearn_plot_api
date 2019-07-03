from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class RocCurveViz:
    def _create(self, est, X, y, response_method="auto", pos_label=None):
        predict_proba = getattr(est, 'predict_proba', None)
        decision_function = getattr(est, 'decision_function', None)

        if response_method == 'auto':
            prediction_method = predict_proba or decision_function
        elif response_method == 'predict_proba':
            prediction_method = predict_proba
        elif response_method == 'decision_function':
            prediction_method = decision_function
        if prediction_method is None:
            raise ValueError("Estimator does not have proper response method")

        y_pred = prediction_method(X)
        if y_pred.ndim == 2:
            y_pred = y_pred[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred, pos_label=pos_label)

        self.fpr_ = fpr
        self.tpr_ = tpr
        self.label_ = est.__class__.__name__
        return self

    def plot(self, ax=None, viz=None):
        if viz is not None:
            ax = viz.ax_
        elif ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        self.line_ = ax.plot(self.fpr_, self.tpr_, label=self.label_)[0]
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def _plot(cls, est, X, y, response_method="auto", pos_label=None, ax=None):
        new_viz = cls()._create(est,
                                X,
                                y,
                                response_method=response_method,
                                pos_label=pos_label)
        new_viz.plot(ax=ax)
        return new_viz


plot_roc_curve = RocCurveViz._plot
