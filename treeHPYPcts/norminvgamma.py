"""
Implements normal inverse gamma (https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution) distribution
as scipy does not currently include it.

@author: Steven Nguyen
"""

from scipy.stats import rv_continuous
from scipy.stats import norm
from scipy.stats import gengamma
from scipy.special import gamma
from scipy.stats import expon
import numpy as np

class norminvgamma():

    def __init__(self, mu0, nu, alpha, beta):
        self.mu0 = mu0
        self.nu = nu
        self.alpha = alpha
        self.beta = beta

    def argcheck(self, mu0, nu, alpha, beta):
        return (self.alpha > 0)

    def rvs(self, size=1):
        sigma_2 = gengamma.rvs(self.alpha, self.beta, size=size)
        sigma_2 = np.array(sigma_2)
        if size == 1:
            return [norm.rvs(self.mu0, sigma_2[0] / self.nu), sigma_2[0]]
        return [[norm.rvs(self.mu0, s / self.nu), s] for s in sigma_2]

    def pdf(self, mu, var):
        t1 = ((self.nu) ** 0.5) * ((self.beta) ** self.alpha)
        t2 = (var * (2 * 3.15) ** 0.5) * gamma(self.alpha)
        t3 = (1 / var ** 2) ** (self.alpha + 1)
        t4 = expon.pdf((2 * self.beta + self.nu * (self.mu0 - mu) ** 2) / (2 * var ** 2))
        # print (t1, t2, t3, t4)
        return (t1 / t2) * t3 * t4


