import logging
import time

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score

from logreg.logistic_regressor import LogisticRegressor


@pytest.mark.parametrize(
    "parameters,batch_size",
    [
        (
            dict(
                penalty="l2",
                alpha=0.001,
                n_iter=100,       # was max_iter
                tol=0.01,
                lr=0.01,          # was eta0
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
            ),
            64,
        ),
        (
            dict(
                penalty="l2",
                alpha=0.002,
                n_iter=100,
                tol=0.01,
                lr=0.02,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=7,
            ),
            64,
        ),
        (
            dict(
                penalty="l1",
                alpha=0.01,
                n_iter=100,
                tol=0.01,
                lr=0.005,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
            ),
            128,
        ),
    ]
)
def test_sklearn_logistic(parameters: dict, batch_size: int):
    rng = np.random.default_rng(123)
    n, d = 10000, 50
    X = rng.normal(size=(n, d))
    true_w = rng.uniform(-2, 2, size=d)
    true_b = rng.uniform(-0.5, 0.5)
    logits = X @ true_w + true_b
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p, size=n).astype(float)

    split = 7500
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # --- your model ---
    my_best_ll = np.inf
    my_best_acc = 0.0
    my_time = 0.0
    for _ in range(30):  # 반복 횟수 축소로 테스트 시간 관리
        model = LogisticRegressor(batch_size=batch_size, **parameters)
        t0 = time.time()
        model.fit(X_train, y_train)
        my_time += time.time() - t0

        proba = model.predict(X_test)
        yhat = (proba >= 0.5).astype(float)
        my_best_ll = min(my_best_ll, log_loss(y_test, proba, labels=[0, 1]))
        my_best_acc = max(my_best_acc, accuracy_score(y_test, yhat))

    # --- sklearn baseline ---
    sk_best_ll = np.inf
    sk_best_acc = 0.0
    sk_time = 0.0
    for _ in range(30):
        sk = SGDClassifier(
            loss="log_loss",
            penalty=parameters["penalty"],
            alpha=parameters["alpha"],
            max_iter=parameters["n_iter"],
            tol=parameters["tol"],
            learning_rate="constant",
            eta0=parameters["lr"],
            random_state=parameters["random_state"],
            early_stopping=parameters["early_stopping"],
            validation_fraction=parameters["validation_fraction"],
            n_iter_no_change=parameters["n_iter_no_change"],
            shuffle=True,
        )
        t0 = time.time()
        sk.fit(X_train, y_train)
        sk_time += time.time() - t0

        proba_sk = sk.predict(X_test)
        yhat_sk = (proba_sk >= 0.5).astype(float)
        sk_best_ll = min(sk_best_ll, log_loss(y_test, proba_sk, labels=[0, 1]))
        sk_best_acc = max(sk_best_acc, accuracy_score(y_test, yhat_sk))

    logging.info(f"Sklearn best logloss: {sk_best_ll:.4f}, acc: {sk_best_acc:.4f}")
    logging.info(f"Your best logloss:  {my_best_ll:.4f}, acc: {my_best_acc:.4f}")
    logging.info(f"Sklearn time: {sk_time:.3f}s, Your time: {my_time:.3f}s")

    # 성능과 시간 비교 기준
    assert my_best_ll <= sk_best_ll * 1.2
    assert my_time <= sk_time * 6
