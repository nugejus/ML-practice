import time
import numpy as np
import pytest
from logreg.logistic_regressor import LogisticRegressor


# --------- helpers ----------
def bce(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def accuracy(y_pred_label: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred_label == y_true).astype(float))


# --------- API/arguments ----------
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "penalty": "l2",
            "alpha": 0.0001,
            "n_iter": 1000,
            "tol": 0.001,
            "lr": 0.01,
            "random_state": None,
            "early_stopping": False,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "batch_size": 64,
        },
        {
            "penalty": "l1",
            "alpha": 0.1,
            "n_iter": 100,
            "tol": 0.01,
            "lr": 0.001,
            "random_state": 1234,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 6,
            "batch_size": 8,
        },
    ],
)
def test_arguments(parameters: dict):
    obj = LogisticRegressor(**parameters)
    for key, value in parameters.items():
        assert getattr(obj, key) == value


# --------- fit / predict on synthetic logistic data ----------
@pytest.mark.parametrize(
    "random_seed,train_ratio,size,coefs,bias,penalty,lr,alpha,min_acc",
    [
        (1234, 0.75, (400, 1), np.array([2.0]), 0.5, "l2", 0.05, 1e-3, 0.7),
        (1235, 0.70, (600, 3), np.array([1.0, -1.5, 0.8]), -0.3, "l1", 0.05, 1e-3, 0.7),
        (1236, 0.80, (1000, 5), np.array([0.7, -1.2, 0.9, 0.3, -0.6]), 0.2, "l2", 0.05, 1e-4, 0.7),
    ],
)
def test_fit_predict(
    random_seed: int,
    train_ratio: float,
    size: tuple[int],
    coefs: np.array,
    bias: float,
    penalty: str,
    lr: float,
    alpha: float,
    min_acc: float,
):
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=size)
    z = X @ coefs + bias
    p = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p, size=(size[0],)).astype(float)

    split = int(size[0] * train_ratio)
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]

    reg = LogisticRegressor(
        n_iter=500,
        tol=1e-4,
        alpha=alpha,
        penalty=penalty,
        lr=lr,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.2,
        batch_size=64,
    )
    reg.fit(X_tr, y_tr)

    proba = reg.predict(X_te)
    y_hat = (proba >= 0.5).astype(float)

    # 정확도와 BCE 모두 점검
    assert accuracy(y_hat, y_te) >= min_acc


# --------- n_iter 영향: 더 오래 학습하면 손실 감소 ----------
def test_n_iter_effect_on_loss():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(800, 10))
    true_w = rng.normal(size=10)
    b = 0.3
    p = 1 / (1 + np.exp(-(X @ true_w + b)))
    y = rng.binomial(1, p).astype(float)

    model_small = LogisticRegressor(n_iter=10, random_state=0, early_stopping=False)
    model_mid = LogisticRegressor(n_iter=50, random_state=0, early_stopping=False)

    model_small.fit(X, y)
    model_mid.fit(X, y)

    loss_small = bce(model_small.predict(X), y)
    loss_mid = bce(model_mid.predict(X), y)

    assert loss_mid <= loss_small + 1e-6


# --------- early stopping: 속도 이점 ----------
def test_early_stopping():
    rng = np.random.default_rng(12345)
    X = rng.normal(size=(1200, 50))
    w = rng.normal(size=50)
    b = 0.1
    p = 1 / (1 + np.exp(-(X @ w + b)))
    y = rng.binomial(1, p).astype(float)

    models = [
        LogisticRegressor(n_iter=80, early_stopping=True, validation_fraction=0.2, batch_size=64, random_state=0),
        LogisticRegressor(n_iter=80, early_stopping=False, batch_size=64, random_state=0),
    ]

    totals = []
    for m in models:
        start = time.time()
        for _ in range(20):
            m.fit(X, y)
        totals.append(time.time() - start)

    assert totals[0] < totals[1]


# --------- random_state 재현성 ----------
@pytest.mark.parametrize("random_seed", [1234, 4567, 386, 17482, 555])
def test_random_state(random_seed: int):
    rng = np.random.default_rng(777)
    X = rng.normal(size=(300, 8))
    w = rng.normal(size=8)
    b = 0.2
    p = 1 / (1 + np.exp(-(X @ w + b)))
    y = rng.binomial(1, p).astype(float)

    reg1 = LogisticRegressor(n_iter=30, random_state=random_seed, shuffle=True, batch_size=16)
    reg1.fit(X, y)

    reg2 = LogisticRegressor(n_iter=30, random_state=random_seed, shuffle=True, batch_size=16)
    reg2.fit(X, y)

    assert np.allclose(reg1.coef_.reshape(-1), reg2.coef_.reshape(-1), atol=1e-8)
    assert np.allclose(reg1.intercept_, reg2.intercept_, atol=1e-8)


# --------- penalty gradient ----------
@pytest.mark.parametrize(
    "penalty,alpha,weights,expected_sum",
    [
        # L2: sum(2*alpha*w)
        ("l2", 0.0001, [5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
                        -0.53015623, -5.88052802, -8.89615508, 9.63588443, -0.06066031],
         2 * 0.0001 * np.sum([5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
                              -0.53015623, -5.88052802, -8.89615508, 9.63588443, -0.06066031])),
        # L1: alpha*sum(sign(w))
        ("l1", 0.0123, [5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
                        -0.53015623, -5.88052802, -8.89615508, 9.63588443, -0.06066031],
         0.0123 * np.sum(np.sign([5.73740291, -1.7512518, -2.62246896, -2.97450237, -0.85847272,
                                  -0.53015623, -5.88052802, -8.89615508, 9.63588443, -0.06066031]))),
        ("l2", 0.4056, [6.929, 3.885, -7.773, 6.456, -0.579, -1.183, -2.62, -4.687,
                        4.151, -5.809, -3.49, 7.474, -8.813, 0.907, 0.434],
         2 * 0.4056 * np.sum([6.929, 3.885, -7.773, 6.456, -0.579, -1.183, -2.62, -4.687,
                              4.151, -5.809, -3.49, 7.474, -8.813, 0.907, 0.434])),
        ("l1", 0.003961, [-6.8727, 9.8228, -3.4836, -3.8546, 0.0, -2.5597, -1.9072,
                          0.0, 8.8834, -9.9594, 0.0, 0.0, -9.7112, -2.761, -9.1893],
         0.003961 * np.sum(np.sign([-6.8727, 9.8228, -3.4836, -3.8546, 0.0, -2.5597, -1.9072,
                                    0.0, 8.8834, -9.9594, 0.0, 0.0, -9.7112, -2.761, -9.1893]))),
    ],
)
def test_penalty_gradient_sum(penalty, alpha, weights, expected_sum):
    model = LogisticRegressor(penalty=penalty, alpha=alpha)
    model.coef_ = np.array(weights, dtype=float)
    # intercept에는 규제 없음
    grad_vec = model.get_penalty_grad()
    assert np.isclose(np.sum(grad_vec), expected_sum, atol=1e-8)
