import numpy as np
from scipy import special as sp_spec
from math import pi, sqrt, cos, sin, acos
import matplotlib.pyplot as plt


class single_mie:
    def __init__(self, a, wl, eps_r, eps_t, eps_m=1.0, mu_r=1.0, mu_t=1.0, mu_m=1.0,
                 method='miepython'):
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

        if method == 'my':
            self.em_sca_coef_my()
        elif method == 'miepython':
            self.em_sca_coef_miepython()

    def em_sca_coef_my(self):
        """
        calculates self.an and self.bn scattering coefficients
        a[k], b[k] correspond to order k+1
        """

        m = np.sqrt(self.eps_t) * np.sqrt(self.mu_t) / (np.sqrt(self.eps_m) * np.sqrt(self.mu_m))
        x = self.k * self.a

        nmax = int(round(x + 4 * x ** (1. / 3.) + 2.))
        self.nmax = np.round(max(nmax, np.abs(m * x)) + 16)

        besx = np.zeros(self.nmax, dtype=np.complex)
        dbesx = np.zeros(self.nmax, dtype=np.complex)
        dbesx2 = np.zeros(self.nmax, dtype=np.complex)
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
            besx[n - 1] = sp_spec.spherical_jn(n, x)
            dbesx[n - 1] = sp_spec.spherical_jn(n, x, True)
            hanx[n - 1] = sqx * sp_spec.hankel1(n + 0.5, x)  # sph. hankel 1st kind
            dhanx[n - 1] = sqx * sp_spec.h1vp(n + 0.5, x) + dsqx * sp_spec.hankel1(n + 0.5, x)  # d. sph. hankel 1st

            n1 = np.sqrt(n * (n + 1) * self.eps_t / self.eps_r + 0.25) - 0.5
            besmx_e[n - 1] = sqmx * sp_spec.jv(n1 + 0.5, m * x)
            dbesmx_e[n - 1] = sqmx * sp_spec.jvp(n1 + 0.5, m * x) + dsqmx * sp_spec.jv(n1 + 0.5, m * x)

            n2 = np.sqrt(n * (n + 1) * self.mu_t / self.mu_r + 0.25) - 0.5
            besmx_m[n - 1] = sqmx * sp_spec.jv(n2 + 0.5, m * x)
            dbesmx_m[n - 1] = sqmx * sp_spec.jvp(n2 + 0.5, m * x) + dsqmx * sp_spec.jv(n2 + 0.5, m * x)

        self.an = ((self.mu_m * m ** 2 * (besx + x * dbesx) * besmx_e - self.mu_t * besx * (
                besmx_e + m * x * dbesmx_e)) /
                   (self.mu_m * m ** 2 * (hanx + x * dhanx) * besmx_e - self.mu_t * hanx * (
                           besmx_e + m * x * dbesmx_e)))
        self.bn = ((self.mu_t * (besx + x * dbesx) * besmx_m - self.mu_m * besx * (besmx_m + m * x * dbesmx_m)) /
                   (self.mu_t * (hanx + x * dhanx) * besmx_m - self.mu_m * hanx * (besmx_m + m * x * dbesmx_m)))

    def em_sca_coef_miepython(self):
        m = np.sqrt(self.eps_t) * np.sqrt(self.mu_t) / (np.sqrt(self.eps_m) * np.sqrt(self.mu_m))
        x = self.k * self.a
        (self.an, self.bn) = _mie_An_Bn(m, x)
        self.nmax = len(self.an)

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
            return n / (n - 1) * self.ratio_of_double_factorials(n - 2)

    def csca_fb(self):
        """
        forward and backward scattering cross sections
        :return:  (Csca_f, Csca_b)
        """

        # first part of the sum
        orders = np.arange(1, self.nmax + 1)
        p1 = np.sum((2 * orders + 1) * (np.abs(self.an) ** 2 + np.abs(self.bn) ** 2))

        # second part of the sum
        kk = orders[1::2, np.newaxis]  # evn (int)
        ll = orders[np.newaxis, 0::2]  # odd (int)

        dfac_ratio_kk = np.array([1 / self.ratio_of_double_factorials(k) for k in kk[:, 0]])[:, np.newaxis]
        dfac_ratio_ll = np.array([self.ratio_of_double_factorials(l) for l in ll[0, :]])[np.newaxis, :]

        p2 = np.sum((-1) ** ((kk + ll - 1) / 2) * (2 * kk + 1) * (2 * ll + 1) / ((kk - ll) * (kk + ll + 1))
                    * dfac_ratio_kk  # (sp_spec.factorial2(kk - 1) / sp_spec.factorial2(kk))
                    * dfac_ratio_ll  # (sp_spec.factorial2(ll) / sp_spec.factorial2(ll - 1))
                    * (self.an[1::2, np.newaxis] * self.an[np.newaxis, 0::2].conjugate()
                       + self.bn[1::2, np.newaxis] * self.bn[np.newaxis, 0::2].conjugate()).real)

        # third part of the sum
        kk = orders[0::2, np.newaxis]

        dfac_ratio_kk = np.array([self.ratio_of_double_factorials(k) for k in kk[:, 0]])[:, np.newaxis]

        p3 = np.sum((-1) ** ((kk + ll) / 2) * (2 * kk + 1) * (2 * ll + 1) / (kk * (kk + 1) * ll * (ll + 1))
                    * dfac_ratio_kk  # (sp_spec.factorial2(kk) / sp_spec.factorial2(kk-1))
                    * dfac_ratio_ll  # (sp_spec.factorial2(ll) / sp_spec.factorial2(ll-1))
                    * (self.an[0::2, np.newaxis] * self.bn[np.newaxis, 0::2].conjugate()).real)

        csca_f = pi / self.k ** 2 * (p1 - 2 * p2 - 2 * p3)
        csca_b = pi / self.k ** 2 * (p1 + 2 * p2 + 2 * p3)

        return csca_f, csca_b

    def cext(self, ):
        return 2 * pi / self.k ** 2 * np.sum(np.real(self.an + self.bn) * (2. * np.arange(1, self.nmax + 1) + 1.))


# the following part of the code is copied from the miepython package:
# https://github.com/scottprahl/miepython
def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.
    Args:
        z: function argument
        N: order of Ricatti-Bessel function
    Returns:
        This returns the Ricatti-Bessel function of order N with argument z
        using the continued fraction technique of Lentz, Appl. Opt., 15,
        668-671, (1976).
    """
    zinv = 2.0 / z
    alpha = (N + 0.5) * zinv
    aj = -(N + 1.5) * zinv
    alpha_j1 = aj + 1 / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while abs(abs(ratio) - 1.0) > 1e-12:
        aj = zinv - aj
        alpha_j1 = 1.0 / alpha_j1 + aj
        alpha_j2 = 1.0 / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        zinv *= -1
        runratio = ratio * runratio

    return -N / z + runratio


def _D_downwards(z, N):
    """
    Compute the logarithmic derivative by downwards recurrence.
    Args:
        z: function argument
        N: order of Ricatti-Bessel function
    Returns:
        All the Ricatti-Bessel function values for orders from 0 to N for an
        argument z using the downwards recurrence relations.
    """
    D = np.zeros(N, dtype=complex)
    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        last_D = n / z - 1.0 / (last_D + n / z)
        D[n - 1] = last_D
    return D


def _D_upwards(z, N):
    """
    Compute the logarithmic derivative by upwards recurrence.
    Args:
        z: function argument
        N: order of Ricatti-Bessel function
    Returns:
        All the Ricatti-Bessel function values for orders from 0 to N for an
        argument z using the upwards recurrence relations.
    """
    D = np.zeros(N, dtype=complex)
    exp = np.exp(-2j * z)
    D[1] = -1 / z + (1 - exp) / ((1 - exp) / z - 1j * (1 + exp))
    for n in range(2, N):
        D[n] = 1 / (n / z - D[n - 1]) - n / z
    return D


def _D_calc(m, x, N):
    """
    Compute the logarithmic derivative using best method.
    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function
    Returns:
        The values of the Ricatti-Bessel function for orders from 0 to N.
    """
    n = m.real
    kappa = abs(m.imag)

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n ** 2:
        return _D_downwards(m * x, N)

    return _D_upwards(m * x, N)


def _mie_An_Bn(m, x):
    """
    Compute arrays of Mie coefficients A and B for a sphere.
    This estimates the size of the arrays based on Wiscombe's formula. The length
    of the arrays is chosen so that the error when the series are summed is
    around 1e-6.
    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
    Returns:
        An, Bn: arrays of Mie coefficents
    """
    nstop = int(x + 4.05 * x ** 0.33333 + 2.0) + 1

    if m.real > 0.0:
        D = _D_calc(m, x, nstop + 1)

    a = np.zeros(nstop - 1, dtype=complex)
    b = np.zeros(nstop - 1, dtype=complex)

    psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
    psi_n = psi_nm1 / x - np.cos(x)  # n = 1
    xi_nm1 = complex(psi_nm1, np.cos(x))
    xi_n = complex(psi_n, np.cos(x) / x + np.sin(x))

    for n in range(1, nstop):
        if m.real == 0.0:
            a[n - 1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
            b[n - 1] = psi_n / xi_n
        else:
            temp = D[n] / m + n / x
            a[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[n] * m + n / x
            b[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)

        xi = (2 * n + 1) * xi_n / x - xi_nm1
        xi_nm1 = xi_n
        xi_n = xi
        psi_nm1 = psi_n
        psi_n = xi_n.real

    return a, b
