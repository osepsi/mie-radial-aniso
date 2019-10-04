import numpy as np
from scipy import special as sp_spec
from math import pi, sqrt, cos, sin, acos
import sys
import time
import matplotlib.pyplot as plt


class single_mie:
    def __init__(self, a, wl, eps_r, eps_t, eps_m=1.0, mu_r=1.0, mu_t=1.0, mu_m=1.0):
        """
        a: radius
        wl: wavelength
        eps_r, eps_t: radial and tangential relative permittivity
        eps_m: background medium relative permittivity
        mu_r, mu_t, mu_m: same as for eps
        """

        self.a = a
        self.wl = wl
        self.eps_r = eps_r
        self.eps_t = eps_t
        self.eps_m = eps_m
        self.mu_r = mu_r
        self.mu_t = mu_t
        self.mu_m = mu_m
        self.n_m = sqrt(self.eps_m) * sqrt(self.mu_m)
        self.k = 2 * pi / self.wl * self.n_m
        self.nmax = None
        self.an = None
        self.bn = None

        self.em_sca_coef()

    def em_sca_coef(self):
        """
        calculates self.an and self.bn scattering coefficients
        a[k], b[k] correspond to order k+1
        """

        m = np.sqrt(self.eps_t) * np.sqrt(self.mu_t) / (np.sqrt(self.eps_m) * np.sqrt(self.mu_m))
        x = self.k * self.a

        self.nmax = int(round(x + 4 * x ** (1. / 3.) + 2.))*2
        besx = np.zeros(self.nmax, dtype=np.complex)
        dbesx = np.zeros(self.nmax, dtype=np.complex)
        hanx = np.zeros(self.nmax, dtype=np.complex)
        dhanx = np.zeros(self.nmax, dtype=np.complex)
        besmx_e = np.zeros(self.nmax, dtype=np.complex)
        dbesmx_e = np.zeros(self.nmax, dtype=np.complex)
        besmx_m = np.zeros(self.nmax, dtype=np.complex)
        dbesmx_m = np.zeros(self.nmax, dtype=np.complex)

        sqx = np.sqrt(0.5 * pi / x)
        dsqx = -0.5 * np.sqrt(0.5 * pi / x ** 3)
        sqmx = np.sqrt(0.5 * pi / (m * x))
        dsqmx = -0.5 * np.sqrt(0.5 * pi / (m * x) ** 3)
        for n in range(1, self.nmax + 1):
            besx[n - 1] = sqx * sp_spec.jv(n + 0.5, x)  # sph bessel 1st kind
            dbesx[n - 1] = sqx * sp_spec.jvp(n + 0.5, x) + dsqx * sp_spec.jv(n + 0.5, x)  # d. sph bessel 1st kind
            hanx[n - 1] = sqx * sp_spec.hankel1(n + 0.5, x)  # sph. hankel 1st kind
            dhanx[n - 1] = sqx * sp_spec.h1vp(n + 0.5, x) + dsqx * sp_spec.hankel1(n + 0.5, x)  # d. sph. hankel 1st

            n1 = np.sqrt(n * (n + 1) * self.eps_t / self.eps_r + 0.25) - 0.5
            besmx_e[n - 1] = sqmx * sp_spec.jv(n1 + 0.5, m * x)
            dbesmx_e[n - 1] = sqmx * sp_spec.jvp(n1 + 0.5, m * x) + dsqmx * sp_spec.jv(n1 + 0.5, m * x)
            # print n1, besmx_e[n-1], dbesmx_e[n-1]

            n2 = np.sqrt(n * (n + 1) * self.mu_t / self.mu_r + 0.25) - 0.5
            besmx_m[n - 1] = sqmx * sp_spec.jv(n2 + 0.5, m * x)
            dbesmx_m[n - 1] = sqmx * sp_spec.jvp(n2 + 0.5, m * x) + dsqmx * sp_spec.jv(n2 + 0.5, m * x)

        self.an = ((self.mu_m * m ** 2 * (besx + x * dbesx) * besmx_e - self.mu_t * besx * (
                    besmx_e + m * x * dbesmx_e)) /
                   (self.mu_m * m ** 2 * (hanx + x * dhanx) * besmx_e - self.mu_t * hanx * (
                               besmx_e + m * x * dbesmx_e)))
        self.bn = ((self.mu_t * (besx + x * dbesx) * besmx_m - self.mu_m * besx * (besmx_m + m * x * dbesmx_m)) /
                   (self.mu_t * (hanx + x * dhanx) * besmx_m - self.mu_m * hanx * (besmx_m + m * x * dbesmx_m)))

    def csca(self):
        return 2 * pi / self.k ** 2 * np.sum(
            (np.abs(self.an) ** 2 + np.abs(self.bn) ** 2) * (2. * np.arange(1, self.nmax + 1) + 1.))

    def ratio_of_double_factorials(self, n):
        """
        n!! / (n-1)!!

        :param n:
        :return:
        """
        if n == 1 or n == 0:
            return 1
        else:
            return n/(n-1)*self.ratio_of_double_factorials(n-2)

    def csca_fb(self):
        """
        forward and backward scattering cross sections
        :return:  (Csca_f, Csca_b)
        """

        # first part of the sum
        print(self.nmax)
        orders = np.arange(1, self.nmax + 1)
        p1 = np.sum((2 * orders + 1) * (np.abs(self.an) ** 2 + np.abs(self.bn) ** 2))

        # second part of the sum
        kk = orders[1::2, np.newaxis]  # evn (int)
        ll = orders[np.newaxis, 0::2]  # odd (int)
        print(kk.shape, ll.shape)

        # calculate ratio of double factorials
        # dfac_ratio_kk = np.where(kk < 100, sp_spec.factorial2(kk - 1) / sp_spec.factorial2(kk),
        #                          1 / np.sqrt(np.pi * kk / 2))

        dfac_ratio_kk = np.array([1/self.ratio_of_double_factorials(k) for k in kk[:, 0]])[:, np.newaxis]
        # dfac_ratio_ll = np.where(ll < 100, sp_spec.factorial2(ll) / sp_spec.factorial2(ll - 1),
        #                          np.sqrt(2 * ll ** 2 / (ll - 1) / np.pi))
        dfac_ratio_ll = np.array([self.ratio_of_double_factorials(l) for l in ll[0, :]])[np.newaxis, :]

        # print(ll)
        # print(dfac_ratio_ll)
        # print(dfac_ratio_ll.shape)

        # plt.figure()
        # plt.plot(kk[:,0], dfac_ratio_kk[:,0])
        # plt.plot(kk[:,0], dfac_ratio_kk2[:,0])
        #
        # plt.figure()
        # plt.plot(ll[0], dfac_ratio_ll[0])
        # plt.plot(ll[0], dfac_ratio_l2[0])

        p2 = np.sum((-1) ** ((kk + ll - 1) / 2) * (2 * kk + 1) * (2 * ll + 1) / ((kk - ll) * (kk + ll + 1))
                    * dfac_ratio_kk  # (sp_spec.factorial2(kk - 1) / sp_spec.factorial2(kk))
                    * dfac_ratio_ll  # (sp_spec.factorial2(ll) / sp_spec.factorial2(ll - 1))
                    * (self.an[1::2, np.newaxis] * self.an[np.newaxis, 0::2].conjugate()
                       + self.bn[1::2, np.newaxis] * self.bn[np.newaxis, 0::2].conjugate()).real)
                       
                       
        p2_uj = 2/np.pi*np.sum((-1) ** ((kk + ll - 1) / 2) * (2 * kk + 1) * (2 * ll + 1) / ((kk - ll) * (kk + ll + 1))
                    # * sp_spec.gamma((ll+2)/2)/sp_spec.gamma((ll+1)/2)
                    * np.exp(sp_spec.gammaln((ll+2)/2)-sp_spec.gammaln((ll+1)/2))
                    # * sp_spec.gamma((kk+1)/2)/sp_spec.gamma((kk+2)/2)
                    * np.exp(sp_spec.gammaln((kk+1)/2)-sp_spec.gammaln((kk+2)/2))
                    * (self.an[1::2, np.newaxis] * self.an[np.newaxis, 0::2].conjugate()
                       + self.bn[1::2, np.newaxis] * self.bn[np.newaxis, 0::2].conjugate()).real)

        # third part of the sum
        kk = orders[0::2, np.newaxis]
        # dfac_ratio_kk = np.where(kk < 100, sp_spec.factorial2(kk) / sp_spec.factorial2(kk - 1),
        #                          np.sqrt(2 * kk ** 2 / (kk - 1) / np.pi))
        dfac_ratio_kk = np.array([self.ratio_of_double_factorials(k) for k in kk[:, 0]])[:, np.newaxis]

        # plt.figure()
        # plt.plot(kk[:,0], dfac_ratio_kk[:, 0])
        # plt.plot(kk[:,0], dfac_ratio_k2[:, 0])

        p3 = np.sum((-1) ** ((kk + ll) / 2) * (2 * kk + 1) * (2 * ll + 1) / (kk * (kk + 1) * ll * (ll + 1))
                    * dfac_ratio_kk  # (sp_spec.factorial2(kk) / sp_spec.factorial2(kk-1))
                    * dfac_ratio_ll  # (sp_spec.factorial2(ll) / sp_spec.factorial2(ll-1))
                    * (self.an[0::2, np.newaxis] * self.bn[np.newaxis, 0::2].conjugate()).real)
                    
        p3_uj = 4/np.pi*np.sum((-1) ** ((kk + ll) / 2) * (2 * kk + 1) * (2 * ll + 1) / (kk * (kk + 1) * ll * (ll + 1))
              
                # * dfac_ratio_kk  # (sp_spec.factorial2(kk) / sp_spec.factorial2(kk-1))
                * np.exp(sp_spec.gammaln((kk+2)/2) - sp_spec.gammaln((kk+1)/2))
                    # * dfac_ratio_ll  # (sp_spec.factorial2(ll) / sp_spec.factorial2(ll-1))
                * np.exp(sp_spec.gammaln((ll+2)/2) - sp_spec.gammaln((ll+1)/2))
                    * (self.an[0::2, np.newaxis] * self.bn[np.newaxis, 0::2].conjugate()).real)

        csca_f = pi / self.k ** 2 * (p1 - 2 * p2 - 2 * p3)
        csca_b = pi / self.k ** 2 * (p1 + 2 * p2 + 2 * p3)
        print(pi / self.k ** 2 *p1, pi / self.k ** 2 *2*p2, pi / self.k ** 2 *2*p2_uj, pi / self.k ** 2 *2*p3, pi / self.k ** 2 *2*p3_uj)

        return csca_f, csca_b

    def cext(self, ):
        return 2 * pi / self.k ** 2 * np.sum(np.real(self.an + self.bn) * (2. * np.arange(1, self.nmax + 1) + 1.))
