#! /usr/bin/env python

import argparse as ap
import os
import sys
import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats

from numpy.linalg import multi_dot

from methods import *

mdot = multi_dot


def get_matrix(beta, alpha):
    """
    Generate a random matrix with condition number kappa = beta / alpha
    """
    A = np.load("data/1.npy") #stats.norm.rvs(size=(n, n))
    n, _ = A.shape

    Q, R = linalg.qr(A)
    S = stats.norm.rvs(size=n)
    S = 10 ** S
    Smin = min(S)
    Smax = max(S)
    S = (S - Smin) / (Smax - Smin)
    S = alpha + S * (beta - alpha)
    A = multi_dot([Q.T, np.diag(S), Q])

    return A


def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("n", type=int, help="number of samples")
    argp.add_argument("-a", "--alpha", type=int, default=1, help="minimum eigenvalue")
    argp.add_argument("-b", "--beta", type=int, default=10, help="maximum eigenvalue")
    argp.add_argument("-i", "--iter", type=int, default=10, help="number of iterations")
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)
    H2LOCAL = 0.25

    #methods = [OLS(), RR(plambda=0.1), RobReg(rho=0.1), BLUP()]
    methods = [OLS(), RR(plambda=0.1), BLUP()]

    corr = dict()
    y_rmse = dict()
    b_rmse = dict()

    for method in methods:
        corr[method.name] = []
        y_rmse[method.name] = []
        b_rmse[method.name] = []

    for i in range(args.iter):

        # simulate samples
        V = get_matrix(args.beta, args.alpha)
        p, p = V.shape
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=V, size=args.n * 2)
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)


        # simulate trait
        beta = np.random.normal(loc=0, scale=1, size=p)
        g = X.dot(beta)
        s2g = np.var(g, ddof=1)
        s2e = s2g * (( 1 / H2LOCAL ) - 1)
        eps = np.random.normal(loc=0, scale=np.sqrt(s2e), size=args.n * 2)
        y = g + eps

        ytrain = y[:args.n]
        Xtrain = X[:args.n]
        ytest = y[args.n:]
        Xtest = X[args.n:]

        for method in methods:
            coef = method.fit(Xtrain, ytrain)
            cor = np.corrcoef(Xtest.dot(coef), ytest)[0, 1]
            corr[method.name].append(cor)
            b_rmse[method.name].append(rmse(coef, beta))
            y_rmse[method.name].append(rmse(ytrain, Xtrain.dot(coef)))

    for method in methods:
        args.output.write("{} Avg BETA RMSE = {:.3f} (sd={:.3f})".format(method.name, \
                                                                 np.mean(b_rmse[method.name]), \
                                                                 np.std(b_rmse[method.name])) + os.linesep)
        args.output.write("{} Avg PRED RMSE = {:.3f} (sd={:.3f})".format(method.name, \
                                                                 np.mean(y_rmse[method.name]), \
                                                                 np.std(y_rmse[method.name])) + os.linesep)
        args.output.write("{} Avg CORR = {:.3f} (sd={:.3f})".format(method.name, \
                                                                 np.mean(corr[method.name]), \
                                                                 np.std(corr[method.name])) + os.linesep)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
