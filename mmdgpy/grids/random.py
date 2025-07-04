import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft

################################################################################


def get_gaussian_process(dim, mu, rho, h, seed=None):
    """Returns the points of an equidistant grid on [0,1]^dim and two
    realizations of a stationary Gaussian random field on this grid using a
    circulant embedding method.

    :param int dim: The dimension of the domain of the Gaussian field.
    :param float mu: The constant mean value of the stationary Gaussian
        field.
    :param rho: The function ρ with c(x1,x2) = ρ(x1-x2) for any points x1,x2
        where c denotes the covariance function of the stationary Gaussian
        field.
    :param float h: The grid width.
    :param seed: The seed for the random number generator. Defaults to None.
    :raises NotImplementedError: If dim != 1.
    :raises LinAlgError: If the circulant embedding matrix is not positive
        semidefinite.
    """
    if dim != 1:
        raise NotImplementedError

    if seed is not None:
        np.random.seed(seed)

    m = int(round(1.0 / h))
    x = np.linspace(0.0, 1.0, m + 1)  # 1d grid

    # first column of covariance matrix and its circulant embedding matrix
    r = rho(x)
    s = np.concatenate((r, np.flip(r[1:-1])))

    s_hat = fft(s)

    if np.less(s_hat, 0.0).any():
        raise np.linalg.LinAlgError(
            "Circulant embedding matrix is not positive semidefinite!"
        )

    eps = np.random.normal(size=2 * m) + 1j * np.random.normal(size=2 * m)
    e_hat = np.sqrt(0.5 * s_hat / m) * eps
    e = fft(e_hat)

    return x, mu + np.real(e[: m + 1]), mu + np.imag(e[: m + 1])


################################################################################


def get_gaussian_aperture(dim, mu, rho, h, dmin=1e-6, file=None, seed=None):
    """Returns aperture functions d1, d2 on [0,1]^(dim-1) that define the
    geometry of a fracture. The aperture functions are created as linear
    spline interpolants from two realizations of a (dim-1)-dimensional
    stationary Gaussian random field on an equidistant grid. The fracture is
    required to have a positive minimum aperture dmin. In order to guarantee
    this, values of the discrete Gaussian field are substituted accordingly
    if necessary.

    :param int dim: The bulk dimension.
    :param float mu: The constant mean value of the stationary Gaussian
        field.
    :param rho: The function ρ with c(x1,x2) = ρ(x1-x2) for any points x1,x2
        where c denotes the covariance function of the stationary Gaussian
        field.
    :param float h: The grid width.
    :param float dmin: The minimum aperture of the fracture. Defaults to
        1e-6.
    :param file: A filename. The value of the Gaussian aperture and the
        corresponding grid are saved to the file if a filename is provided. 
        If the filename ends with ".csv", the data is saved in CSV format,
        otherwise in NumPy's .npz format. By default no such file is created.
    :param seed: The seed for the random number generator. Defaults to None.
    :raises NotImplementedError: If dim != 2.
    """
    if dim != 2:
        raise NotImplementedError

    xh, z1, z2 = get_gaussian_process(dim - 1, mu, rho, h, seed=seed)

    overlap = np.where((z1 + z2) <= dmin)

    if np.size(overlap) != 0:
        print(
            f"Warning: Substituting {np.size(overlap)} points with total "
            f"aperture below dmin={dmin}."
        )
        correction = 0.5 * (dmin - z1[overlap] - z2[overlap])
        z1[overlap] += correction
        z2[overlap] += correction

    if file is not None:
        if str(file).lower().endswith('.csv'):
            data = np.column_stack((xh, z1, z2))
            header = "xh,d1,d2"
            np.savetxt(file, data, delimiter=",", header=header, comments='')
        else:
            np.savez(file, xh=xh, d1=z1, d2=z2)

    d1 = interp1d(xh, z1)
    d2 = interp1d(xh, z2)

    return d1, d2
