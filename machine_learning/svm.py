

from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data
X = X[:, [2,3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.svm import SVC
# Linear SVM
model = SVC(kernel='linear', C=1.0, random_state=123)
# Linear kernel: no feature mapping is apllied  (identity mapping f(x) = x).
# The model learns a decision boundary directly in the original feature space.
# Decision function: f(x) = w^T * x + b (a linear hyperplane).
model.fit(X_train, y_train)


import matplotlib.pyplot as plt
from ml_lib.fig_func import plot_decision_regions

plot_decision_regions(
    X=X_train,
    y=y_train,
    classifier=model
)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()

plot_decision_regions(
    X=np.concatenate([X_train, X_test]),
    y=np.concatenate([y_train, y_test]),
    classifier=model, test_idx=range(105, 150)  # last 30% was set to test data
)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()


# Non-Linear SVM
model2 = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=123)     ## good?
# model2 = SVC(kernel='rbf', C=1.0, gamma=10, random_state=123)       ## over-fit
# model2 = SVC(kernel='rbf', C=1.0, gamma=0.01, random_state=123)     ## under-fit
"""
C:     Regularization parameter (inverse of regularization strength).
       This parameter controls the trade-off between maximizing the margin and minimizing classification errors.
           - Smaller C : stronger regularization (simpler model, wider margin).
           - Larger C : weaker regularization (fit training data more closely).

gamma: Kernel coefficient for RBF kernel.
       This parameter controls the influence range of each training sample.
           - Smaller gamma : broader influence (smoother decision boundary).
           - Larger gamma : narrower influence (more complex, localized decision boundary).
"""

model2.fit(X_train, y_train)

plot_decision_regions(
    X=X_train,
    y=y_train,
    classifier=model2
)
plt.xlabel('pental length')
plt.ylabel('petal width')
# plt.legend()
plt.show()

