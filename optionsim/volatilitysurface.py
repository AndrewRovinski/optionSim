
import numpy as np
from dataclasses import dataclass

from .rbf import RadialBasisInterpolation

# ToDo: for stochastic/variable volatility, make the arrays 3d


@dataclass
class VolatilitySurface(object):
    """
    A class representation of a volatility surface at a point in time.
    The volatility surface exists over three dimensions: (Percent Strike,
    Time To Maturity, and Implied Volatility).

    The volatility surface class also has built-in methods to interpolate
    any value on the grid boundaries, given the X and Y coordinates.

    """
    ttm: np.array
    strikes: np.array
    vols: np.ndarray

    def __post_init__(self):

        self.ttm, self.strikes = np.meshgrid(self.ttm, self.strikes)

        self.interpolator = RadialBasisInterpolation(
            np.array([self.ttm.ravel(), self.strikes.ravel()]).T,
            self.vols.ravel(), kernel_type='linear')

        # Scipy's Radial Basis Function gives bad results for flat/sparse grids

        # self.interpolator = Rbf(self.ttm, self.strikes, self.vols,
        #                         method='thin-plate', smooth=0.5)

    def iv(self, ttm, strike):
        # return self.interpolator(ttm, strike)
        return self.interpolator(np.array([float(ttm), strike]).T)

    def iv_grid(self, ttm, strikes):
        return self.interpolator(np.array([ttm, strikes]).T)

