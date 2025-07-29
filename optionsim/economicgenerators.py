#!/usr/bin/env python3
"""
ECONOMIC SCENARIO GENERATORS

"""

import math
import pandas as pd
import numpy as np
from fractions import Fraction
from abc import ABC, abstractmethod

from .volatilitysurface import VolatilitySurface


class EconomicGenerator(ABC):

    @abstractmethod
    def generate_path(self, proj_length: int, time_delta: Fraction) -> any:
        pass


class LinearEquityPathGenerator(EconomicGenerator):
    """
    Base class for a deterministic (constant drift) equity scenario
    generator. Allows for specification of an index name and initial
    level such that multiple indices can be added to a Scenario.
    """

    def __init__(self, drift_rate: float, index_name: str = 'equity',
                 index_level: float = 100) -> None:
        """
        :param index_name: string name of the index
        :param index_level: float of the index level
        :param drift_rate: float of the *ARITHMETIC* drift rate
        """

        self.drift = drift_rate
        self.index_name = index_name
        self.index_level = index_level

    def generate_path(self, proj_length: int,
                      time_delta: Fraction) -> pd.DataFrame:
        """
        Function to generate a DataFrame of equity returns.

        :param proj_length: int of the total projection periods
        :param time_delta: fraction of the period time step
        :return: DataFrame of the equity returns at each time step
        """

        index_level_str = f'{self.index_name}_level'
        index_return_str = f'{self.index_name}_returns'
        path = pd.DataFrame()
        returns = np.power((1 + self.drift), time_delta) - 1
        path[index_return_str] = [returns] * int(proj_length / time_delta)
        path.loc[-1] = [0]
        path.index = path.index + 1
        path = path.sort_index()
        levels = self.index_level * (1 + path[index_return_str]).cumprod()
        path[index_level_str] = levels
        return path


class GeometricBrownianMotionGenerator(EconomicGenerator):
    """
    Base class for generating index levels and returns that follow a
    Geometric Brownian Motion.

    NOTE: Inputs for the drift rate are GEOMETRIC rates!
    """

    def __init__(self, drift_rate: float, volatility: float,
                 index_name: str = 'equity', index_level: float = 100,
                 seed: int = None) -> None:
        """
        :param drift_rate: float of the *GEOMETRIC* equity drift rate
        :param volatility: float of the volatility (vol of log-returns)
        :param index_name: string of the index name
        :param index_level: float of the initial index level
        :param seed: integer of the seed to use in the random number generator
        """

        self.drift = drift_rate
        self.vol = volatility
        self.index_name = index_name
        self.index_level = index_level
        self.seed = seed

    def generate_path(self, proj_length: int,
                      time_delta: Fraction) -> pd.DataFrame:
        """
        Function to generate a DataFrame of equity returns.

        :param proj_length: int of the total projection periods
        :param time_delta: float of the period fractions to use as time steps
        :return: DataFrame of the equity returns at each time step
        """

        index_level_str = f'{self.index_name}_level'
        path = pd.DataFrame()
        n_steps = int(proj_length / time_delta)

        np.random.seed(self.seed)

        x = np.random.normal(
            (self.drift - 0.5 * self.vol**2) * time_delta,
            self.vol * math.sqrt(time_delta),
            n_steps)

        x = np.cumsum(x)
        levels = self.index_level * np.exp(x)
        path[index_level_str] = levels
        path.loc[-1] = self.index_level
        path.index = path.index + 1
        path = path.sort_index()
        return path


# ToDo: Extend outputs to 3d object (time, maturity, rate)
class FlatYieldCurveGenerator(EconomicGenerator):
    """
    Generates a series of continuous rates
    """

    def __init__(self, continuous_rate: float) -> None:
        """
        :param continuous_rate: float of the zero rate
        """

        self.rate = continuous_rate

    def generate_path(self, proj_length: int,
                      time_delta: Fraction) -> pd.DataFrame:
        """
        Function to generate a DataFrame of interest rates.

        :param proj_length: int of the total projection periods
        :param time_delta: float of the period fractions to use as time steps
        :return: DataFrame of the interest rate at each time step
        """

        path = pd.DataFrame()
        steps = int(proj_length / time_delta) + 1
        path['interest_rates'] = [self.rate] * steps
        return path


class FlatDividendYieldGenerator(EconomicGenerator):
    """
    Generates a constant time series of dividend yields
    """

    def __init__(self, continuous_div_yield: float) -> None:
        """
        :param continuous_div_yield: float of the dividend yield
        """

        self.div_yield = continuous_div_yield

    def generate_path(self, proj_length: int,
                      time_delta: Fraction) -> pd.DataFrame:
        """
        Function to generate a DataFrame of dividend yields.

        :param proj_length: int of the total projection periods
        :param time_delta: float of the period fractions to use as time steps
        :return: DataFrame of the dividend yield at each time step
        """

        path = pd.DataFrame()
        steps = int(proj_length / time_delta) + 1
        path['dividend_yield'] = [self.div_yield] * steps

        return path


class FlatVolatilitySurfaceGenerator(EconomicGenerator):
    """
    Generates a pandas DataFrame of implied volatility surface objects.

    NOTE: This generator takes a singular IV figure and generates a flat
    3d volatility surface; as opposed to the 'StaticVolatilitySurfaceGenerator'
    which takes a grid of IV values, and does not vary through time.
    """

    def __init__(self, expirations: np.ndarray, percent_strikes: np.ndarray,
                 volatility: float) -> None:
        """
        :param volatility: float of the volatility
        :param percent_strikes: numpy array of the percent strikes to use
        :param expirations: numpy array of the year fractions (as Fractions)
        """

        self.expirations = expirations
        self.percent_strikes = percent_strikes
        self.volatility = volatility

    def generate_path(self, proj_length: int,
                      time_delta: Fraction) -> pd.DataFrame:
        """
        Generates a series of single volatility values

        :param proj_length: int of the total projection periods
        :param time_delta: float of the period fractions to use as time steps
        :return: DataFrame of the volatility at each time step
        """

        vols = np.zeros((self.expirations.shape[0],
                        self.percent_strikes.shape[0]))

        vols[...] = self.volatility
        surf = VolatilitySurface(self.expirations, self.percent_strikes, vols)
        steps = int(proj_length / time_delta) + 1
        path = pd.DataFrame()
        path['implied_volatility'] = [surf] * steps

        return path


class StaticVolatilitySurfaceGenerator(object):
    """
    Generates a static volatility surface. Unlike the FlatVolatilitySurface,
    the StaticVolatilitySurface allows for volatility skew and term structure.
    """

    def __init__(self, expirations: np.ndarray, percent_strikes: np.ndarray,
                 volatility: np.ndarray) -> None:
        """
        :param volatility: numpy array of the implied volatility values
        :param percent_strikes: numpy array of the percent strikes to use
        :param expirations: numpy array of the year fractions (as Fractions)
        """

        self.expirations = expirations
        self.percent_strikes = percent_strikes
        self.volatility = volatility

    def generate_path(self, proj_length: int,
                      time_delta: Fraction) -> pd.DataFrame:
        """
        Generates a series of single volatility values

        :param proj_length: int of the total projection periods
        :param time_delta: float of the period fractions to use as time steps
        :return: DataFrame of the volatility at each time step
        """

        surf = VolatilitySurface(
            self.expirations,
            self.percent_strikes,
            self.volatility)

        steps = int(proj_length / time_delta) + 1
        path = pd.DataFrame()
        path['implied_volatility'] = [surf] * steps

        return path
