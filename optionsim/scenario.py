
import numpy as np
import pandas as pd
from fractions import Fraction
from dataclasses import asdict

from .economicgenerators import EconomicGenerator
from .volatilitysurface import VolatilitySurface
from .timeslice import TimeSlice


# ToDo: Implement DateTime indexed scenarios instead of time steps/fractions
# ToDo: Vectorize generation of paths, find way to "glue" rate/vol objects
class Scenario(object):
    """
    A Singleton class for constructing the scenarios to use in the simulation.
    """

    def __init__(self, proj_length: int, time_delta: Fraction, n_paths: int,
                 equity_gen: EconomicGenerator = None,
                 div_gen: EconomicGenerator = None,
                 rate_gen: EconomicGenerator = None,
                 vol_gen: EconomicGenerator = None) -> None:
        """
        :param proj_length: length of the projection period (years)
        :param time_delta: time step for scenarios (year fractions)
        :param equity_gen: equity path generator object
        :param div_gen: dividend yield generator object
        :param rate_gen: interest rate generator object
        :param vol_gen: implied volatility surface generator object
        """

        self.proj_length = proj_length
        self.time_delta = time_delta
        self.n_paths = n_paths
        self.equity_gen = equity_gen
        self.div_gen = div_gen
        self.rate_gen = rate_gen
        self.vol_gen = vol_gen
        self.paths = {}

    def from_files(self, file_map: dict) -> None:
        """
        Takes the given file mapping dictionary and reads the CSV data.

        :param file_map: dict of the 'variable': 'filepath' mapping
        :return: None, loads the CSV data into self.paths
        """

        equity = pd.read_csv(file_map['spx_level'])
        dividends = pd.read_csv(file_map['dividend_yield'])
        rates = pd.read_csv(file_map['interest_rates'])
        vols = pd.read_csv(file_map['implied_volatility'], header=None)

        # Create the volatility surface object
        expirations = np.array(vols.iloc[0].dropna())
        strikes = np.array(vols.iloc[:, 0].dropna())
        values = np.array(vols.iloc[1:, 1:])
        surface = VolatilitySurface(expirations, strikes, values)

        for path_num in range(1, self.n_paths + 1):

            # ToDo: fix issues with equity level names
            path = pd.DataFrame()
            path['spx_level'] = equity[f'spx_level_{path_num}']
            path['dividend_yield'] = dividends[f'dividend_yield_{path_num}']
            path['interest_rates'] = rates[f'interest_rates_{path_num}']
            path['time_step'] = path.index
            path['time_delta'] = self.time_delta
            path.loc[0, 'time_delta'] = 0
            path['periodic_yield'] = (path['interest_rates'].shift(1)
                                      * self.time_delta)

            path.loc[0, 'periodic_yield'] = 0

            # DataFrame column will be an array of references
            path['implied_volatility'] = surface

            slices = path.apply(self._to_slices, axis=1)
            self.paths[path_num] = slices

    def generate(self, save_paths: bool = False,
                 file_map: dict = None) -> None:
        """
        Generates scenario paths with the given generators.

        NOTE: the scenario DataFrame contains REFERENCES to the volatility
        surface object instance - the rows do not contain instances!!

        :param save_paths: boolean flag to save generated paths to CSV outputs
        :param file_map: dict of the 'variable': 'filepath' mappings
        :return: None
        """

        for path_num in range(1, self.n_paths + 1):

            # Generate the equity level
            equity = self.equity_gen.generate_path(
                self.proj_length, self.time_delta)

            # Generate the interest rates
            rates = self.rate_gen.generate_path(
                self.proj_length, self.time_delta)

            # Generate the dividend yield
            dividends = self.div_gen.generate_path(
                self.proj_length, self.time_delta)

            # Generate the volatility surface
            vols = self.vol_gen.generate_path(
                self.proj_length, self.time_delta)

            # Concatenate the variables into a single dataframe
            path = pd.concat([equity, rates, dividends, vols], axis=1)

            # Add columns for the time step number and time delta
            path['time_step'] = path.index
            path['time_delta'] = self.time_delta
            path.loc[0, 'time_delta'] = 0

            # Add periodic yield
            path['periodic_yield'] = rates.shift(1) * self.time_delta
            path.loc[0, 'periodic_yield'] = 0

            slices = path.apply(self._to_slices, axis=1)
            self.paths[path_num] = slices

        if save_paths:

            for variable, filepath in file_map.items():
                out_file = pd.DataFrame()
                for path_num, path in self.paths.items():
                    column_name = f'{variable}_{path_num}'
                    path = path.apply(asdict)
                    path = pd.DataFrame(path.tolist())
                    out_file[column_name] = path[variable]

                out_file.to_csv(filepath, index=False)

    # ToDo: Speed this up
    @staticmethod
    def _to_slices(row: pd.Series) -> TimeSlice:
        """
        Private method to create a TimeSlice object from a DataFrame row

        :param row: pd.Series object (like the kind used in pd.apply
        :return: TimeSlice object
        """

        time_slice = TimeSlice(
            time_step=row['time_step'],
            time_delta=row['time_delta'],
            spx_level=row['spx_level'],
            dividend_yield=row['dividend_yield'],
            interest_rates=row['interest_rates'],
            implied_volatility=row['implied_volatility'],
            periodic_yield=row['periodic_yield'])

        return time_slice
