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
        return

    def fit(self, X, y, coef):
        pass

    @property
    def beta_errors(self):
        return self._beta_errors

    @property
    def pred_errors(self):
        return self._pred_errors

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
            step = prj.project_onto_chi_square_ball(step, self.rho)

            loss_v = loss(X.dot(step), y)
            if np.abs(loss_v - last) < 1e-6:
                break
            last = loss_v

        self.beta_errors.append(rmse(step, coef))
        self.pred_errors.append(rmse(X.dot(step), y))
        return

    @property
    def name(self):
        return "Variance-regularized regression with rho = {}".format(self.rho)

class OLS(Method):
    def fit(self, X, y, coef):
        Vest = np.corrcoef(X.T)
        bhat = mdot([linalg.pinv(Vest), X.T, y])
        self.beta_errors.append(rmse(bhat, coef))
        self.pred_errors.append(rmse(X.dot(bhat), y))
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
        return

    @property
    def name(self):
        return "Ridge Regression (lambda ={})".format(self.plambda)
