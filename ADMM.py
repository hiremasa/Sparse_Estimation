""" ADMM lasso
"""

import numpy as np
import numba
import gc
import logging

# gc設定
gc.enable()
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(format=fmt)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ADMM:
    def __init__(self, X, y, lamb, rho=1):
        self.X = X
        self.y = y
        self.l = lamb
        self.rho = rho
        self.N, self.p = self.X.shape

        self.inv_ = self.X.T @ self.y
        self.y_tilde = np.linalg.inv(self.X.T @ self.X + self.N * self.rho * np.eye(self.p))

        self.beta = self.X.T @ self.y
        self.gamma = self.beta.copy()
        self.u = np.zeros(self.p)

        self.variable_history_list = []
        self.regularization_changed_flag = []

    def solve(self, max_iteration=50, tol=1e-3, message=True):
        converged = False

        for iteration_index in range(max_iteration):
            self.variable_history_list.append(self.return_variables())
            logger.debug(self.return_variables())

            pre_beta = self.beta
            self._update_beta()
            self._update_gamma()
            self._update_u()

            diff = max(np.abs(self.beta - pre_beta))
            if diff < tol and iteration_index > 3:
                converged = True
                if message:
                    logger.info("converged!!")
                    logger.debug(f"diff = {diff}")
                    logger.debug(f"iteration num = {iteration_index + 1}")
                break

        if not converged:
            logger.info("doesn't converged")
            logger.debug(f"diff = {diff}")

    def _update_beta(self):
        self.beta = self.inv_ @ (self.y_tilde + self.N * self.rho * (self.gamma - self.u / self.rho))

    def _update_gamma(self):
        self.gamma = soft_threshold(self.beta + self.u / self.rho, self.l / self.rho)

    def _update_u(self):
        self.u = self.u + self.rho * (self.beta - self.gamma)

    def return_variables(self):
        return [
            self.beta.mean(),
            self.gamma.mean(),
            self.u.mean()
        ]

def soft_threshold(x, lamb):
    return np.sign(x) * np.maximum(np.abs(x) - lamb, 0)



if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    # Generate some sparse data to play with
    np.random.seed(42)

    n_samples, n_features = 50, 200
    X = np.random.randn(n_samples, n_features)
    coef = 3 * np.random.randn(n_features)
    inds = np.arange(n_features)
    np.random.shuffle(inds)
    coef[inds[10:]] = 0  # sparsify coef
    y = np.dot(X, coef)

    # add noise
    y += 0.01 * np.random.normal(size=n_samples)

    # std
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = ADMM(X, y, lamb=30)
    model.solve()
