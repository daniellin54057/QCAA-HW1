##	1.	import
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

import matplotlib.pyplot as plt
import time
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set a random seed
np.random.seed(13202050)


def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    Xvals, yvals = [], []

    for _ in range(samples):
        x = 2 * np.random.rand(2) - 1
        y = 1 if np.linalg.norm(x - center) < radius else 0
        Xvals.append(x)
        yvals.append(y)

    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)


def moons(samples=200, noise=0.1, seed=13202050):
    X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    return np.array(X, requires_grad=False), np.array(y, requires_grad=False)


def to_pm_one(y):
    # 0/1 -> -1/+1
    return np.array(2 * y - 1, requires_grad=False)

# ===== 3. data re-uploading model =====
dev_reupload = qml.device("default.qubit", wires=1)


@qml.qnode(dev_reupload, interface="autograd")
def reupload_circuit(x, params):
    # params shape = (n_layers, 3)
    for p in params:
        qml.Rot(x[0], x[1], 0.0, wires=0)   # data re-uploading
        qml.Rot(p[0], p[1], p[2], wires=0)  # trainable layer
    return qml.expval(qml.PauliZ(0))


class DataReuploadingClassifier:
    def __init__(self, n_layers=3, steps=100, lr=0.1):
        self.n_layers = n_layers
        self.steps = steps
        self.lr = lr
        self.params = 0.01 * np.random.randn(n_layers, 3, requires_grad=True)
        self.training_time = None

    def predict_scores(self, X):
        return np.array([reupload_circuit(x, self.params) for x in X], requires_grad=False)

    def predict(self, X):
        scores = self.predict_scores(X)
        return np.array((scores > 0).astype(int), requires_grad=False)

    def loss(self, params, X, y_pm):
        preds = np.array([reupload_circuit(x, params) for x in X])
        return np.mean((preds - y_pm) ** 2)

    def fit(self, X, y):
        y_pm = to_pm_one(y)
        opt = AdamOptimizer(self.lr)

        start = time.perf_counter()
        params = self.params

        for step in range(self.steps):
            params = opt.step(lambda v: self.loss(v, X, y_pm), params)

            if (step + 1) % 20 == 0:
                current_loss = self.loss(params, X, y_pm)
                print(f"[Reupload] step {step+1:3d} | loss = {current_loss:.4f}")

        self.params = params
        self.training_time = time.perf_counter() - start

    def num_params(self):
        return self.n_layers * 3
##	4.	explicit model
##	5.	implicit kernel model
##	6.	training / evaluation functions
##	7.	decision boundary plotting
##	8.	Fig. 6 analogue plotting
##	9.	comparison table
# ===== Dataset test =====
X_circle, y_circle = circle(200)
X_moons, y_moons = moons(200, noise=0.1)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_circle, y_circle, test_size=0.3, random_state=42, stratify=y_circle
)

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42, stratify=y_moons
)

print("Circle:", Xc_train.shape, Xc_test.shape)
print("Moons:", Xm_train.shape, Xm_test.shape)

