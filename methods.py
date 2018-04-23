import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats

from numpy.linalg import multi_dot

import projection as prj

mdot = multi_dot


class Method(object):

    def __init__(self):
        return

    def fit(self, X, y):
        pass

class RobReg(Method):

    def __init__(self, rho=0.1):
        self.rho = rho
        super(RobReg, self).__init__()
        return

    def fit(self, X, y):

        def loss(pred, ground):
            return (pred - ground) ** 2

        last = -1
        loss_v = -1
        n, p = X.shape
        bhat = np.zeros(p)
        ypred = X.dot(bhat)

        eta = 0.5
        c = 0.5
        tau = 0.5

        while True:
            pstar = prj.project_onto_chi_square_ball(loss(ypred, y), self.rho)
            P = np.diag(pstar)
            bhat = mdot([linalg.inv(mdot([X.T, P, X])), X.T, P, y])
            ypred = X.dot(bhat)

            loss_v = np.sum(pstar * loss(ypred, y))
            if np.abs(loss_v - last) < 1e-6:
                break
            last = loss_v

        return bhat

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

    def fit(self, X, y):
        C, V, = self._reml(X, y)
        bhat = mdot([C, linalg.inv(V), y])
        return bhat

    @property
    def name(self):
        return "BLUP"


class OLS(Method):
    def fit(self, X, y):
        Vest = np.corrcoef(X.T)
        bhat = mdot([linalg.pinv(Vest), X.T, y])
        return bhat

    @property
    def name(self):
        return "Psuedo-inverse OLS"


class RR(Method):
    def __init__(self, plambda=0.1):
        self.plambda = plambda
        super(RR, self).__init__()
        return

    def fit(self, X, y):
        n, p = X.shape
        Vest = np.corrcoef(X.T)
        bhat = mdot([linalg.inv(Vest + np.eye(p) * self.plambda), X.T, y])
        return bhat

    @property
    def name(self):
        return "Ridge Regression (lambda ={})".format(self.plambda)
