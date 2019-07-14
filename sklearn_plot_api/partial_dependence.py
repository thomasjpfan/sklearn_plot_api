from sklearn.inspection import partial_dependence
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np


class PartialDependenceViz:
    def _create(self, est, X, features, *, feature_names, response_method,
                percentiles, grid_resolution, method, n_jobs):
        # regression only for now
        pd_results = Parallel(n_jobs=n_jobs)(
            delayed(partial_dependence)(est,
                                        X,
                                        feature,
                                        response_method=response_method,
                                        method=method,
                                        grid_resolution=grid_resolution,
                                        percentiles=percentiles)
            for feature in features)
        self.pd_results_ = pd_results
        self.feature_names_ = feature_names
        return self

    def plot(self, ax=None, n_cols=2):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            fig = ax.figure

        pd_results_ = self.pd_results_
        n_features = len(pd_results_)
        feature_names_ = self.feature_names_

        n_cols = min(n_cols, n_features)
        n_rows = int(np.ceil(n_features / float(n_cols)))

        gs = GridSpecFromSubplotSpec(n_cols,
                                     n_rows,
                                     subplot_spec=ax.get_subplotspec())

        axes = {}
        ax.set_subplotspec(gs[0])
        ax.update_params()
        ax.set_position(ax.figbox)
        axes[feature_names_[0]] = ax

        for i in range(1, n_features):
            feature_name = feature_names_[i]
            axes[feature_name] = fig.add_subplot(gs[i])

        artists = {}
        for feature_name, (avg_preds, values) in zip(feature_names_,
                                                     pd_results_):
            cur_ax = axes[feature_name]
            artist = cur_ax.plot(values[0], avg_preds[0].ravel())[0]
            artists[feature_name] = artist
            cur_ax.set_xlabel(feature_name)

        self.axes_ = axes
        self.artists_ = artists
        self.figure_ = ax.get_figure()
        return self

    @classmethod
    def _plot(cls,
              est,
              X,
              features,
              feature_names,
              response_method='auto',
              percentiles=(0.05, 0.95),
              grid_resolution=100,
              method='auto',
              n_jobs=None,
              ax=None,
              n_cols=2):
        viz = cls()._create(est,
                            X,
                            features,
                            feature_names=feature_names,
                            response_method=response_method,
                            percentiles=percentiles,
                            grid_resolution=grid_resolution,
                            method=method,
                            n_jobs=n_jobs)
        return viz.plot(ax=ax, n_cols=2)


plot_partial_dependence = PartialDependenceViz._plot
