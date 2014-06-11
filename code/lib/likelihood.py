import pkg_resources
pkg_resources.require("scipy>0.13")

import numpy as np
import pyexp
from math import log, pi, exp, isnan, sqrt
import scipy.optimize
import scipy.misc
import scipy.stats
import scipy.integrate
import warnings
import logging
import psutil


from nearpd import nearPD


def logfac(x):
    return scipy.special.gammaln(x + 1)


def diagmat(M):
    return np.diag(M.diagonal())


def inv_qf(M, x):
    # if __debug__: print("inv_qf")
    # Inverse quadratic form: x^T(M^-1)x
    # More stable than computing M^-1
    v, W = np.linalg.eigh(M)
    s = sum(np.dot(W[:, i], x).item()**2 / v[i] for i in range(len(x)) if v[i] > 0)
    return s


def latent_mvnorm_ll_is(x, mu, sigma, sampling):
    K = len(x)
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        evs = sorted(np.linalg.eigvals(sigma))
        cn = abs(evs[-1] / evs[0])
        warnings.warn("Sigma is not PSD; condition number: %g" % cn)
        sigma = nearPD(sigma)

    # Importance sampler
    counts = x * sampling
    a0 = logfac(sampling) - logfac(counts) - logfac(sampling - counts)
    # Prior distribution
    mvn = scipy.stats.multivariate_normal(mu, sigma)
    NS = 20000
    Z = mvn.rvs(NS)
    splx = np.all([Z > 0, Z < 1], axis=(0, 2))
    Z = Z[splx]
    s = (a0 + counts * np.log(Z) + (sampling - counts) * np.log(1. - Z)).sum(axis=1)
    val = scipy.misc.logsumexp(s) - log(splx.sum())
    return val


def latent_mvnorm_ll_nquad(x, mu, sigma, sampling):
    K = len(x)
    counts = x * sampling
    mvn = scipy.stats.multivariate_normal(mu, sigma)
    a0 = logfac(sampling) - logfac(counts) - logfac(sampling - counts)
    def f(*args):
        Z = np.array(args)
        return exp((a0 + counts * np.log(Z) + (sampling - counts) * np.log(1. - Z)).sum() + mvn.logpdf(Z))
    ret = scipy.integrate.nquad(f, ((0.0, 1.0),) * K)
    print(ret)
    return ret[0]


def latent_mvnorm_ll_laplace(x, mu, sigma, sampling):
    K = len(x)
    sgn, logdetsigma = np.linalg.slogdet(sigma)
    if sgn != 1.0:
        evs = sorted(np.linalg.eigvals(sigma))
        cn = abs(evs[-1] / evs[0])
        warnings.warn("Sigma is not PSD; condition number: %g" % cn)

    # Importance sampler
    counts = x * sampling
    logfac = lambda x: scipy.special.gammaln(x + 1)
    # expo += mvn.logpdf(Z)
    # expo -= prop.logpdf(Z)
    # ret = scipy.misc.logsumexp(expo) - log(splx.sum())
    # Prior distribution
    mvn = scipy.stats.multivariate_normal(mu, sigma)
    sigmainv = np.linalg.inv(sigma)
    a0 = logfac(sampling) - logfac(counts) - logfac(sampling - counts)

    def f(Z):
        return (a0 + counts * np.log(Z) + (sampling - counts) * np.log(1. - Z)).sum()
    def g(Z): 
        Z0 = np.minimum(Z, 0)
        Z1 = np.maximum(Z - 1, 0)
        return -(f(Z) + mvn.logpdf(Z)) + 5 * 0.5 * (np.dot(Z0, Z0) + np.dot(Z1, Z1))
    def grad(Z):
        Z0 = np.minimum(Z, 0)
        Z1 = np.maximum(Z - 1, 0)
        return -((counts / Z) - (sampling - counts) / (1. - Z) - np.dot(sigmainv, Z - mu)) + 5 * (Z0 + Z1)
                 
    xopt, fopt, gopt, Bopt, _, _, _ = scipy.optimize.fmin_bfgs(g, x, grad, full_output=True, gtol=1e-5)
    print(xopt)
    print(mu)
    print(x)
    print(fopt, f(xopt), mvn.logpdf(xopt))
    return -fopt + (K / 2) * log(2 * pi) - (1 / 2) * np.linalg.slogdet(Bopt)[1]

latent_mvnorm_ll = latent_mvnorm_ll_is

def mvnorm_ll(x, mu, sigma, mask):
    'Log-likelihood function of N(mu, sigma) rv.'
    # if __debug__: print("mvnorm_ll")
    # const * -(1/2) * (x - mu)^T (sigma^-1) (x - mu)
    if mask:
        bdry = np.logical_and(x >= 0.1, x <= 0.9)
        if not np.any(bdry):
            return 0.0
        x = x[bdry]
        mu = mu[bdry]
        sigma = sigma[np.ix_(bdry, bdry)]
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        evs = sorted(np.linalg.eigvals(sigma))
        cn = abs(evs[-1] / evs[0])
        warnings.warn("Sigma is not PD; condition number: %g" % cn)
        sigma = nearPD(sigma)
    mvn = scipy.stats.multivariate_normal(mu, sigma)
    return mvn.logpdf(x)

class Likelihood:
    def __init__(self, subsets, times, positions, selected_position, sampling,
            neutral_approx=True, skip_selected_site=False, verbose=False,
            reset=True):
        self.verbose = verbose
        self.reset = reset
        self.selected_position = selected_position
        self.skip_selected_site = skip_selected_site
        self.neutral_approx = neutral_approx
        self.subsets = subsets
        self.sampling = sampling
        self.times = np.array(times, dtype=np.int32)
        self.positions = np.array(positions, dtype=np.int32)
        self.n_replicates = subsets.data.shape[-1]
        T = len(self.times)
        L = len(self.positions)
        self.M = np.zeros([self.n_replicates, T, L], dtype=np.double)
        self.C = np.zeros([self.n_replicates, T, L, T, L], dtype=np.double)

    def mean_cov(self, N, r, h, s):
        if self.reset or psutil.phymem_usage().percent > 90.0:
            pyexp.reset()
        # if __debug__: print("mean_cov", N, r, h, s)
        # Update mean and covariance matrices in place. Returns None.
        for i in range(self.n_replicates):
            if self.neutral_approx and (s == 0 or self.selected_position is None):
                pyexp.mean_cov_neutral(self.M[i], self.C[i], self.positions,
                                  self.subsets.init_haps2, self.times, N, r)

            else:
                pyexp.mean_cov(self.M[i], self.C[i], self.positions, 
                        list(self.positions).index(self.selected_position), 
                        self.subsets.init_haps, self.times, N, r, h, s,
                        self.skip_selected_site)
        # if __debug__: print("exit mean_cov")


    def likelihood(self, N, r, h, s):
        self.mean_cov(N, r, h, s)
        # pyexp.cacheStats()
        # if __debug__:
        #     print("mean matrix:")
        #     print(self.M[0])
        # Construct data mask to skip selected site if requested
        ipos = list(range(len(self.positions)))
        itimes = list(range(len(self.times)))
        dinds = np.ix_(itimes, ipos)
        ddinds = np.ix_(itimes, ipos, itimes, ipos)
        M = self.M[0][dinds]
        C = self.C[0][ddinds]
        dd = C.shape[0] * C.shape[1]
        dtas = self.subsets.data[dinds]
        if self.sampling:
            lmv = [latent_mvnorm_ll(dtas[..., i].reshape(dd), M.reshape(dd), C.reshape(dd, dd), self.sampling) 
                   for i in range(self.n_replicates)]
            return sum(lmv)
        else:
            mv = [mvnorm_ll(dtas[..., i].reshape(dd), M.reshape(dd), C.reshape(dd, dd), True) 
                  for i in range(self.n_replicates)]
            return sum(mv)


    def print_data(self):
        for i in range(self.subsets.data.shape[-1]):
            print(self.subsets.data[..., i])

    def maxlik(self, **kwargs):
        '''Determine maximum likelihood over selection parameter.'''
        if self.verbose: 
            self.print_data()
        def negll(x):
            if self.verbose:
                print(x)
            N = kwargs.get('N', x)
            log10r = kwargs.get('log10r', x)
            h = kwargs.get('h', x)
            s = round(kwargs.get('s', x), 3)
            return -self.likelihood(N, 10**log10r, h, s)
        return scipy.optimize.minimize_scalar(negll, bounds=kwargs['bounds'], method='bounded', tol=kwargs['tol'],
                options={'disp': True})
