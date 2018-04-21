"""simple_projections.py
A small module for solving a distributionally robust optimization on
the chi-square divergence ball on the simplex. The bisection algorithm
takes a desired (relative) solution tolerance as a parameter, and is
extremely quick for reasonable values of n and tolerance.
 Given a n-dimensional vector w and a positive number rho, solves
 minimize_p   .5 * norm(p - w, 2)^2
   s.t.      sum(p) = 1, p >= 0,
             (1/nn) * .5 * sum_{i=1}^n (n * p[i] - 1)^2  <=  rho.
"""

import numpy as np

"""
 p = project_onto_chi_square_ball(w, rho, tol = 1e-10)
 Solves the projection problem given above by bisecting on the dual problem
 maximize_{lam >= 0}  min_{p}  .5 * norm(p - w, 2)^2
                                 - lam * (rho + .5 - .5 * n * norm(p)^2)
                        s.t.    sum(p) = 1, p >= 0
 where we used (1/nn) * .5 * sum_{i=1}^n (n * p[i] - 1)^2 = .5 * (n * norm(p)^2 - 1)
 and duality. The KKT conditions of the inner minimization problem are given by
 p(lam) = (1 / (1 + lam * n)) * max(w - eta, 0)
 where eta is the dual variable for the constraint sum(p) = 1. We
 solve eta such that sum(p(lam)) = 1 in solve_inner_eta.
 Given such eta, first note that the gradient of the dual objective
 g(lam) = min_{p}  .5 * norm(p - w, 2)^2
                     - lam * (rho + .5 - .5 * n * norm(p)^2)
            s.t.    sum(p) = 1, p >= 0
 with respect to lam is given by

 g'(lam) = - (rho + .5 - .5 * n * norm(p(lam))^2).
 Since g is concave, g' is decreasing in lam. Hence, we bisect to find
 the optimal lam:
 If g'(lam) > 0,  i.e. .5 * n * norm(p(lam))^2 > rho + .5,  increase lam
 If g'(lam) < 0,  i.e. .5 * n * norm(p(lam))^2 < rho + .5,  decrease lam.
 -------------- obtaining an finite upper bound for lam^* ---------------
 Note that the optimal dual solution lam^* satisfies
 .5 * n * (1/ (1 + lam * n)^2) * sum (w[i] - eta)^2 = rho + .5
 so that this gives th bound
 (1 + lam^* * n)^2 <= maximum(w)^2 * n^2 / (2 * rho + 1)
 or equivalently,
 lam^* <= (1/n) * (n * maximum(w) / sqrt(2 * rho + 1) - 1) := lam_max.
 ------------------------------------------------------------------------
"""

def project_onto_chi_square_ball(w, rho, tol = 1e-10):
  assert (rho > 0)
  rho = float(rho)

  # sort in decreasing order
  w_sort = np.sort(w) # increasing
  w_sort = w_sort[::-1] # decreasing

  w_sort_cumsum = w_sort.cumsum()
  w_sort_sqr_cumsum = np.square(w_sort).cumsum()
  nn = float(w_sort.shape[0])

  lam_min = 0.0
  lam_max = (1/nn) * (nn * w_sort[0] / np.sqrt(2. * rho + 1.) - 1.)
  lam_init_max = lam_max

  if (lam_max <= 0): # optimal lambda is 0
    (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, 0., rho)
    p = w - eta
    low_inds = p < 0
    p[low_inds] = 0.
    return p

  # bisect on lambda to find the optimal lambda value
  while (lam_max - lam_min > tol * lam_init_max):
    lam = .5 * (lam_max + lam_min)
    (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)

    # compute norm(p(lam))^2 * (1+lam * nn)^2
    thresh = .5 * nn * (w_sort_sqr_cumsum[ind] - 2. * eta * w_sort_cumsum[ind] + eta**2 * (ind+1.))
    if (thresh > (rho + .5) * (1 + lam * nn)**2):
      # constraint infeasible, increase lam (dual var)
      lam_min = lam
    else:
      # constraint loose, decrease lam (dual var)
      lam_max = lam

  lam = .5 * (lam_max + lam_min)
  (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)
  p = w - eta
  low_inds = p < 0
  p[low_inds] = 0
  return (1. / (1. + lam * nn)) * p


"""solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)
 Given lam, solves the optimization problem
 minimize_{p}  .5 * norm(p - w, 2)^2
                     - lam * (rho + .5 - .5 * n * norm(p)^2)
     s.t.      sum(p) = 1, p >= 0
 by solving for eta that satifies sum(p(lam)) = 1 where
 p(lam) = (1 / (1 + lam * n)) * max(w - eta, 0).
 Here, eta is the dual variable for sum(p) = 1. Let w_sort be a sorted
 version of w in decreasing order. Plugging the above equation into
 sum(p) = 1, we obtain
 eta = (1/I) (sum_{i=1}^I w_sort[i] - (1 + lam * n))   ...  (*)
 where I = max{i: w_sort[i] >= eta}. Hence, it suffices to solve for
 I to solve for eta. To this end, define
 f(j) = w_sort[j] - (1/j) * (sum_{i=1}^j w_sort[i] - (1 + lam * n)).
 Then, we have that I = max{j : f(j) >= 0} from which eta can be
 computed as in (*). The function returns the tuple (eta, I).
"""

def solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho):
  fs = w_sort - (w_sort_cumsum - (1. + lam * nn)) / (np.arange(nn) + 1.)
  ind = (fs > 0).sum()-1
  return ((1 / (ind+1.)) * (w_sort_cumsum[ind] - (1. + lam * nn)), ind)
