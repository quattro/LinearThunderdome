import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats

from numpy.linalg import multi_dot

import projection as prj

mdot = multi_dot


def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))


class Method(object):

    def __init__(self):
        self._beta_errors = []
        self._pred_errors = []
        self._corr = []
        return

    def fit(self, X, y, coef):
        pass

    @property
    def beta_errors(self):
        return self._beta_errors

    @property
    def pred_errors(self):
        return self._pred_errors

    @property
    def corr(self):
        return self._corr
    

class RobReg(Method):

    def __init__(self, rho=0.1):
        self.rho = rho
        super(RobReg, self).__init__()
        return

    def fit(self, X, y, coef):

        def loss(pred, ground):
            return linalg.norm(pred - ground) ** 2

        last = -1
        loss_v = -1
        while True:
            step = mdot([linalg.inv(X.T.dot(X)), X.T, y])
            bhat = prj.project_onto_chi_square_ball(step, self.rho)

            loss_v = loss(X.dot(bhat), y)
            if np.abs(loss_v - last) < 1e-6:
                break
            last = loss_v

        self.beta_errors.append(rmse(bhat, coef))
        self.pred_errors.append(rmse(X.dot(bhat), y))
        self.corr.append(np.corrcoef(X.dot(bhat), y)[0, 1])
        return

    @property
    def name(self):
        return "Variance-regularized regression with rho = {}".format(self.rho)

class BLUP(Method):

    def _reml(self, X, y, method="AI"):
        y = np.copy(y)

        y -= y.mean()
        y /= y.std()

        n = len(y)
        A = X.dot(X.T)
        I = np.eye(n)

        sigs = np.array([0.5, 0.5])
        grad = np.zeros(2)
        hess = np.zeros((2, 2))

        ll = -1
        V = sigs[0] * A + sigs[1] * I
        max_iter = 100

        for _ in range(max_iter):
            Vinv = np.linalg.inv(V)
            yVinv = y.dot(Vinv)
            VinvA = Vinv.dot(A)

            grad[0] = -0.5 * np.trace(VinvA) + 0.5 * mdot([yVinv, A, yVinv.T])
            grad[1] = -0.5 * np.trace(Vinv) + 0.5 * mdot([yVinv, yVinv.T])

            if method == "AI":
                # ai-reml
                hess[0, 0] = 0.5 * mdot([yVinv, A, VinvA, yVinv.T])
                hess[0, 1] = 0.5 * mdot([yVinv, A, Vinv, yVinv.T])
                hess[1, 0] = hess[0, 1]
                hess[1, 1] = 0.5 * mdot([yVinv, Vinv, yVinv])
            else:
                # hessian
                hess[0, 0] = 0.5 * np.trace(mdot([VinvA, VinvA])) - 0.5 * mdot([yVinv, A, VinvA, yVinv.T])
                hess[0, 1] = 0.5 * np.trace(mdot([VinvA, Vinv])) - 0.5 * mdot([yVinv, A, Vinv, yVinv.T])
                hess[1, 0] = hess[0, 1]
                hess[1, 1] = 0.5 * np.trace(mdot([Vinv, Vinv])) - 0.5 * mdot([yVinv, Vinv, yVinv.T])

            sigs = sigs + linalg.inv(hess).dot(grad)

            # project back if necessary
            if any(sigs < 0):
                sigs[sigs < 0] = 1e-6

            V = sigs[0] * A + sigs[1] * I
            newll = stats.multivariate_normal.logpdf(y, cov=V)
            if np.abs(newll - ll) < 2e-6:
                break
            ll = newll

        return (sigs[0] * X.T, V)

    def fit(self, X, y, coef):
        C, V, = self._reml(X, y)
        bhat = mdot([C, linalg.inv(V), y])
        self.beta_errors.append(rmse(bhat, coef))
        self.pred_errors.append(rmse(X.dot(bhat), y))
        self.corr.append(np.corrcoef(X.dot(bhat), y)[0, 1])
        return

    @property
    def name(self):
        return "BLUP"


class OLS(Method):
    def fit(self, X, y, coef):
        Vest = np.corrcoef(X.T)
        bhat = mdot([linalg.pinv(Vest), X.T, y])
        self.beta_errors.append(rmse(bhat, coef))
        self.pred_errors.append(rmse(X.dot(bhat), y))
        self.corr.append(np.corrcoef(X.dot(bhat), y)[0, 1])
        return

    @property
    def name(self):
        return "Psuedo-inverse OLS"


class RR(Method):
    def __init__(self, plambda=0.1):
        self.plambda = plambda
        super(RR, self).__init__()
        return

    def fit(self, X, y, coef):
        n, p = X.shape
        Vest = np.corrcoef(X.T)
        bhat = mdot([linalg.inv(Vest + np.eye(p) * self.plambda), X.T, y])
        self.beta_errors.append(rmse(bhat, coef))
        self.pred_errors.append(rmse(X.dot(bhat), y))
        self.corr.append(np.corrcoef(X.dot(bhat), y)[0, 1])
        return

    @property
    def name(self):
        return "Ridge Regression (lambda ={})".format(self.plambda)
