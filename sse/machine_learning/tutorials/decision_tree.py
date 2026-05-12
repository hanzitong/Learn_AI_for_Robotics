

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
X = X[:, [2,3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1
)


from sklearn.tree import DecisionTreeClassifier, plot_tree

model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=123)
model.fit(X_train, y_train)


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from fig_func import plot_decision_regions
plot_decision_regions(
    X=X_train,
    y=y_train,
    classifier=model
)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()


# plot test & training data together
plot_decision_regions(
    X=np.concatenate([X_train, X_test]),
    y=np.concatenate([y_train, y_test]),
    classifier=model,
    test_idx=range(105,150)
)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()


plt.figure(figsize=(10,8))
plot_tree(model)
plt.show()


# # Accuracy
# print("Train accuracy: ", model.score(X_train, y_train))
# print("Train accuracy: ", model.score(X_test, y_test))

# # Confusion Matrix
# from sklearn.metrics import confusion_matrix
# y_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# # Overall Report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

