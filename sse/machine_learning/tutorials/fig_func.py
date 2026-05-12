
import numpy as np

from typing import Optional, Sequence
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt



def plot_decision_regions(
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        classifier: ClassifierMixin,
        test_idx: Optional[Sequence[int]] = None,
        resolution: float = 0.02,
) -> None:
    markers: tuple[str, ...] = ('s', 'x', 'o', '^' 'v')
    colors: tuple[str, ...] = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min: float = X[:, 0].min() -1
    x1_max: float = X[:, 0].max() + 1
    x2_min: float = X[:, 1].min() - 1
    x2_max: float = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    grid_points: NDArray[np.floating] = np.array([xx1.ravel(), xx2.ravel()]).T
    Z: NDArray[np.integer] = classifier.predict(grid_points)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha = 0.8,
            c=colors[idx],
            marker=markers[idx],
            label=str(cl),
        )

    if test_idx is not None:
        X_test: NDArray[np.floating] = X[test_idx, :]
        y_test: NDArray[np.integer] = y[test_idx]

        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            alpha = 1.0,
            linewidth=1,
            marker='o',
            edgecolor='black',
            s=100,
            label='test set',
        )




