from itertools import product

import numpy as np
import matplotlib.pyplot as plt


class GridSearchViz:
    def _create(self, est, params):
        # 2d for now
        assert len(params) == 2

        x_name, y_name = params

        x_params = est.param_grid[x_name]
        y_params = est.param_grid[y_name]

        x_mask_array = est.cv_results_["param_{}".format(x_name)]
        y_mask_array = est.cv_results_["param_{}".format(y_name)]
        test_scores = est.cv_results_["mean_test_score"]

        n_x_params, n_y_params = len(x_params), len(y_params)

        scores = np.zeros((n_y_params, n_x_params), dtype=np.float)

        for i, j in product(range(n_x_params), range(n_y_params)):
            x_param = x_params[i]
            y_param = y_params[j]

            x_mask = x_mask_array == x_param
            y_mask = y_mask_array == y_param
            scores[j, i] = np.mean(test_scores[x_mask & y_mask])

        self.scores_ = scores
        self.x_label_ = x_name
        self.y_label_ = y_name
        self.x_params_ = x_params
        self.y_params_ = y_params
        return self

    def plot(self, ax=None, cmap='viridis', include_values=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        scores_ = self.scores_
        im = ax.imshow(scores_, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(scores_.shape[1]),
               yticks=np.arange(scores_.shape[0]),
               xticklabels=self.x_params_,
               yticklabels=self.y_params_,
               ylabel=self.x_label_,
               xlabel=self.y_label_)
        if include_values:
            thresh = scores_.max() / 2.
            for i in range(scores_.shape[0]):
                for j in range(scores_.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(scores_[i, j], '.2f'),
                        ha="center",
                        va="center",
                        color="white" if scores_[i, j] < thresh else "black")
        self.ax_ = ax
        self.figure_ = ax.figure
        self.im_ = im
        return self

    @classmethod
    def _plot(cls, est, params, cmap='viridis', include_values=False):
        viz = cls()._create(est, params)
        return viz.plot(cmap=cmap, include_values=include_values)


plot_grid_search = GridSearchViz._plot
