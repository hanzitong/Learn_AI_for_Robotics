


from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data
X = X[:, [2,3]]
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=1, random_state=123, solver='lbfgs', multi_class='ovr')
lr = LogisticRegression(C=5, random_state=123, solver='lbfgs')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
sum(y_pred == y_test) / len(y_test)
print(lr.coef_)
print(lr.intercept_)



from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from fig_func import plot_decision_regions

plot_decision_regions(
    X=np.concatenate([X_train, X_test]),
    y=np.concatenate([y_train, y_test]),
    classifier=lr,
    test_idx=range(105,150)
)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()


