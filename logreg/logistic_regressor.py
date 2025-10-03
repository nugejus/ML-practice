import numpy as np
from typing import Optional

class LogisticRegressor:
    def __init__(
        self,
        lr: float = 0.1,
        n_iter: int = 1000,
        penalty="l2",
        alpha=0.0001,
        tol: float = 1e-6,
        batch_size: Optional[int] = 64,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        validation_fraction: float =0.1,
        early_stopping : bool = True,
        n_iter_no_change=5,
    ):
        
        self.lr = lr
        self.n_iter = n_iter
        self.penalty = penalty
        self.alpha = alpha
        self.tol = tol
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        self._rand = np.random.default_rng(self.random_state)

        self._w = None
        self._b = 0.0

        self.mean_ = None
        self.std_ = None

    def _std(self, x):
        return (x - self.mean_) / self.std_
    
    def _calc(self, x):
        return x @ self._w + self._b
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, x, y):
        p = self._sigmoid(self._calc(x))
        return -1/len(x) * np.sum(y*np.log(p) + (1 - y)*np.log(1-p))

    def get_penalty_grad(self):
        """Return penalty gradient term based on penalty type."""
        if self.penalty == "l2":
            return 2.0 * self.alpha * self._w
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._w)
        elif not self.penalty or self.penalty == "none":
            return 0.0
        else:
            raise RuntimeError
        
    def _gradient(self, xb, p):
        return (xb.T @ p) / len(xb) + self.get_penalty_grad(), p.mean()
    
    def _gd(self, xb, yb):
        p = self._sigmoid(self._calc(xb)) - yb

        grad_w, grad_b = self._gradient(xb, p)

        self._w -= self.lr * grad_w
        self._b -= self.lr * grad_b

    def predict(self, x):
        x = self._std(x)
        return self._sigmoid(self._calc(x))

    def fit(self, x, y):
        self.mean_ = x.mean(axis = 0)
        self.std_ = x.std(axis = 0)

        x = self._std(x)

        n, d = x.shape

        self._w = self._rand.normal(size = d)
        if self.validation_fraction < 1.0:
            m = int(np.floor(n * (1.0 - self.validation_fraction)))
            idx = np.arange(n)
            self._rand.shuffle(idx)
            tr_idx, va_idx = idx[:m], idx[m:]
            x_tr, y_tr = x[tr_idx], y[tr_idx]
            x_va, y_va = x[va_idx], y[va_idx]
        else:
            x_tr, y_tr = x, y
            x_va, y_va = None, None

        best_val = None
        no_change = 0
        best_w, best_b = self._w.copy(), self._b
        n = len(x_tr)
        for _ in range(self.n_iter):
            if self.shuffle:
                idx = np.arange(n)
                self._rand.shuffle(idx)

            bs = max(1, min(self.batch_size, len(x_tr)))
            for start in range(1, len(x_tr), bs):
                sel = idx[start:start+bs]
                xb = np.take(x_tr, sel, axis=0)
                yb = np.take(y_tr, sel, axis=0)
                self._gd(xb, yb)

            if self.early_stopping:
                val_loss = self._loss(x_va,y_va)
                if (best_val is None) or (
                    val_loss < best_val - self.tol * max(1, best_val)
                ):
                    best_val = val_loss
                    no_change = 0
                    best_w, best_b = self._w.copy(), self._b
                else:
                    no_change += 1
                    if no_change > self.n_iter_no_change:
                        self._w, self._b = best_w, best_b
                        break


    @property
    def coef_(self):
        """Return model coefficients."""
        return self._w

    @property
    def intercept_(self):
        """Return model intercept."""
        return self._b

    @coef_.setter
    def coef_(self, value):
        """Set model coefficients."""
        self._w = value

    @intercept_.setter
    def intercept_(self, value):
        """Set model intercept."""
        self._b = value
