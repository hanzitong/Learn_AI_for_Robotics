
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# MNIST を読み込み
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# 計算時間を短くするため、一部だけ使う
n_samples = 2000
X = X[:n_samples]
y = y[:n_samples].astype(int)

# スケーリング
X = StandardScaler().fit_transform(X)

# t-SNE による 2 次元圧縮
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    max_iter=1000
)
X_2d = tsne.fit_transform(X)

# 可視化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10)
plt.colorbar(scatter, ticks=range(10), label="digit")
plt.title("t-SNE visualization of MNIST")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("../fig/mnist_tsne.png")
plt.show()






