import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def not_(x, y=None):
    return 1 - x


def false_(a, b):
    return 0 * a * b


def true(a, b):
    return not_(false_(a, b))


def or_(a, b):
    return a | b


def xor_(a, b):
    return (a + b) % 2


def and_(a, b):
    return a & b


def nand_(a, b):
    return not_(and_(a, b))


def nor_(a, b):
    return not_(or_(a, b))


def imp_to_b(a, b):
    if a == 1 and b == 0:
        return 0
    return 1


def not_imp_to_b(a, b):
    return not_(imp_to_b(a, b))


def a_(a, b):
    return a


def b_(a, b):
    return b


def not_a_(a, b):
    return not_(a_(a, b))


def not_b(a, b):
    return not_(b_(a, b))


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolor="black",
        )


if __name__ == "__main__":
    algos = [
        Perceptron(),
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        SVC(kernel="rbf"),
        SVC(kernel="linear"),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=2),
        KNeighborsClassifier(n_neighbors=3),
        MLPClassifier(
            hidden_layer_sizes=(2, 3, 2),
            activation="tanh",
            learning_rate="adaptive",
            learning_rate_init=0.01,
            max_iter=2000,
        ),
    ]

    for j, algo in enumerate(algos):
        print(algo)
        plt.figure(figsize=(20, 20), constrained_layout=True)
        plt.suptitle(algo)
        for _, f in enumerate([or_, xor_, and_, nand_, nor_, imp_to_b, not_imp_to_b]):
            X = np.array([[a, b] for a in [0, 1] for b in [0, 1]])
            y = np.array([f(a, b) for a in [0, 1] for b in [0, 1]])
            model = algo
            model.fit(X, y)
            y_pred = model.predict(X)
            print(f.__name__, accuracy_score(y, y_pred))
            x = plt.subplot(3, 3, _ + 1)
            x.set_title(f.__name__)
            plot_decision_regions(X, y, classifier=model)
        plt.savefig(f"lab3/{str(algo).split('(')[0]}.png", dpi=400)
