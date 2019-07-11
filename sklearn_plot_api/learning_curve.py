import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection._split import check_cv
from sklearn.model_selection import learning_curve


class LearningCurveViz:
    def _create(self, estimator, X, y, train_sizes=None, cv=None, n_jobs=None):
        if train_sizes is None:
            train_sizes = np.linspace(.1, 1.0, 5)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        self.train_sizes_ = train_sizes
        self.train_scores_mean_ = np.mean(train_scores, axis=1)
        self.train_scores_std_ = np.std(train_scores, axis=1)
        self.test_scores_mean_ = np.mean(test_scores, axis=1)
        self.test_scores_std_ = np.std(test_scores, axis=1)

        return self

    def plot(self, cmap='tab10', ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        cmap = plt.get_cmap(cmap)
        self.train_fill_between_ = ax.fill_between(
            self.train_sizes_,
            self.train_scores_mean_ - self.train_scores_std_,
            self.train_scores_mean_ + self.train_scores_std_,
            alpha=0.1,
            color=cmap(0))
        self.test_fill_between_ = ax.fill_between(
            self.train_sizes_,
            self.test_scores_mean_ - self.test_scores_std_,
            self.test_scores_mean_ + self.test_scores_std_,
            alpha=0.1,
            color=cmap(1))
        self.train_line_ = ax.plot(self.train_sizes_,
                                   self.train_scores_mean_,
                                   'o-',
                                   color=cmap(0),
                                   label='Training score')[0]
        self.test_line_ = ax.plot(self.train_sizes_,
                                  self.test_scores_mean_,
                                  'o-',
                                  color=cmap(1),
                                  label='Cross-validation score')[0]
        ax.legend(loc='best')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
        ax.grid()

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def _plot(cls,
              estimator,
              X,
              y,
              train_sizes=None,
              cv=None,
              n_jobs=1,
              ax=None,
              cmap='tab10'):
        cv = check_cv(cv)
        plotter = cls()._create(estimator,
                                X,
                                y,
                                train_sizes=train_sizes,
                                cv=cv,
                                n_jobs=n_jobs)
        plotter.plot(ax=ax, cmap=cmap)
        return plotter


plot_learning_curve = LearningCurveViz._plot
