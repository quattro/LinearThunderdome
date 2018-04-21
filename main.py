#! /usr/bin/env python

import argparse as ap
import os
import sys
import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats

from numpy.linalg import multi_dot

from methods import *
import projection as prj

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

    methods = [OLS(), RR(plambda=0.1), RobReg(rho=1.0)]
    for i in range(args.iter):

        # simulate samples
        V = get_matrix(args.beta, args.alpha)
        p, p = V.shape
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=V, size=args.n)
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)


        # simulate trait
        beta = np.random.normal(loc=0, scale=1, size=p)
        g = X.dot(beta)
        s2g = np.var(g, ddof=1)
        s2e = s2g * (( 1 / H2LOCAL ) - 1)
        eps = np.random.normal(loc=0, scale=np.sqrt(s2e), size=args.n)
        y = g + eps

        for method in methods:
            method.fit(X, y, beta)

    for method in methods:
        args.output.write("{} Avg BETA RMSE = {} (sd={})".format(method.name, \
                                                                 np.mean(method.beta_errors), \
                                                                 np.std(method.beta_errors)) + os.linesep)
        args.output.write("{} Avg PRED RMSE = {} (sd={})".format(method.name, \
                                                                 np.mean(method.pred_errors), \
                                                                 np.std(method.pred_errors)) + os.linesep)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
