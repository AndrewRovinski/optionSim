
import numpy as np
from fractions import Fraction
from typing import Callable
from abc import ABC, abstractmethod

from .timeslice import TimeSlice
from .utils import CachedProperty


class Asset(ABC):
    """Abstract Base Class for hedge assets"""

    quantity: int or float
    time_slice: TimeSlice
    type: str

    @property
    @abstractmethod
    def value(self):
        pass


class VanillaOption(Asset):
    """Base class implementation for vanilla european equity put options."""

    # Class ID counter to use for individual instance ID's
    _id = 1

    def __init__(self, option_type: str, underlying: str, strike: float,
                 ttm: Fraction, pricing_model: Callable,
                 quantity: int, time_slice: TimeSlice) -> None:
        """
        :param option_type: string of the type of option (call or put)
        :param underlying: string of the underlying asset for the option
        :param strike: float of the strike price of the option
        :param ttm: Fraction of the time to maturity (in year fractions)
        :param quantity: int of the option position
        :param pricing_model: function that accepts pricing parameters
        """

        self.option_type = option_type
        self.underlying = underlying
        self.strike = strike
        self.ttm = ttm
        self.pricing_model = pricing_model
        self._quantity = quantity
        self._time_slice = time_slice
        self.type = f'{self.__class__.__name__} {self.option_type}'
        self.id = self._id
        self.__class__._id += 1

    @property
    def spot(self) -> float:
        return self.time_slice[f'{self.underlying}_level']

    @property
    def rfr(self) -> float:
        return self.time_slice['interest_rates']

    @property
    def div_yield(self) -> float:
        return self.time_slice['dividend_yield']

    @property
    def sigma(self) -> float:
        pct_strike = self.strike / self.spot
        iv = self.time_slice.implied_volatility.iv(self.ttm, pct_strike)
        return iv.item()

    @property
    def quantity(self) -> int:
        return self._quantity

    @quantity.setter
    @CachedProperty.invalidate('value', 'shocked_value')
    def quantity(self, value: int) -> None:
        self._quantity = value

    @property
    def time_slice(self) -> TimeSlice:
        return self._time_slice

    @time_slice.setter
    @CachedProperty.invalidate('value', 'shocked_value')
    def time_slice(self, sliceobj: TimeSlice) -> None:
        self._time_slice = sliceobj
        self.ttm -= sliceobj['time_delta']

    @CachedProperty
    def value(self) -> float:
        """Memoized property that caches values where possible."""
        if self.ttm > 0:
            option_value = self.pricing_model(
                self.option_type,
                self.spot,
                self.strike,
                float(self.ttm),
                self.rfr,
                self.div_yield,
                self.sigma)

        elif self.ttm == 0:
            if self.option_type == 'put':
                option_value = max(self.strike - self.spot, 0)
            elif self.option_type == 'call':
                option_value = max(self.spot - self.strike, 0)
            else:
                raise ValueError("'option_type' must be 'put' or 'call'!")
        else:
            raise ValueError(f'ERROR: ttm {self.ttm} < 0!')
        return option_value * self.quantity

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'option_type={repr(self.option_type)}, '
            f'underlying={repr(self.underlying)}, '
            f'strike={repr(self.strike)}, '
            f'ttm={repr(self.ttm)}, '
            f'quantity={repr(self.quantity)}, '
            f'pricing_model={repr(self.pricing_model)}, '
            f'value={repr(self.value)}, '
            f'id={repr(self.id)})')


class EquityFuture(Asset):
    """
    Base class for futures

    NOTE: Futures are assumed to have a value of $0 at each time step due
    to mark-to-market cash flows. The 'shocked value' of the contract is
    the difference between the shocked notional and the current notional.
    P&L over the lifetime of the contract can be calculated from the
    current vs. open notional value. The 'mtm' attribute is a placeholder
    to calculate the change in notional between time steps for cash flows.
    """

    # Class ID counter to use for individual instance ID's
    _id = 1

    def __init__(self, underlying: str, ttm: Fraction, quantity: float,
                 time_slice: TimeSlice, multiplier: int = 50):
        """
        :param underlying: string of the underlying index
        :param ttm: float of the time to maturity (in year fractions)
        :param quantity: int of the number of contracts
        :param time_slice: TimeSlice instance to use for valuation
        :param multiplier: int of the contract multiplier
        """

        self.underlying = underlying
        self.ttm = ttm
        self.quantity = quantity
        self._time_slice = time_slice
        self.multiplier = multiplier
        self.mtm = None
        self.type = f'{self.__class__.__name__}'
        self.id = self._id
        self.__class__._id += 1

    @property
    def time_slice(self) -> TimeSlice:
        return self._time_slice

    @time_slice.setter
    def time_slice(self, sliceobj: TimeSlice) -> None:
        previous_notional = self.notional
        self._time_slice = sliceobj
        self.ttm -= sliceobj['time_delta']
        self.mtm = self.notional - previous_notional

    @property
    def spot(self) -> float:
        return self.time_slice[f'{self.underlying}_level']

    @property
    def rfr(self) -> float:
        return self.time_slice['interest_rates']

    @property
    def div_yield(self) -> float:
        return self.time_slice['dividend_yield']

    @property
    def value(self) -> int:
        return 0

    @property
    def notional(self) -> float:
        f_price = self.spot * np.exp((self.rfr - self.div_yield) * self.ttm)
        return f_price * self.multiplier * self.quantity

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'underlying={repr(self.underlying)}, '
            f'ttm={repr(self.ttm)}, '
            f'quantity={repr(self.quantity)}, '
            f'multiplier={repr(self.multiplier)}, '
            f'id={repr(self.id)}, '
            f'notional={repr(self.notional)})')


class SyntheticPut(Asset):
    _id = 1

    def __init__(self, underlying: str, call_strike: float, ttm: Fraction,
                 pricing_model: Callable, quantity: int,
                 time_slice: TimeSlice, multiplier: int = 50):

        self.underlying = underlying
        self.call_strike = call_strike
        self.ttm = ttm
        self.pricing_model = pricing_model
        self._quantity = quantity
        self._time_slice = time_slice
        self.multiplier = multiplier
        self.type = f'{self.__class__.__name__}'
        self.id = self._id
        self.__class__._id += 1

        self.call = self._build_call()
        self.future = self._build_future()

    def _build_call(self):
        call_option = VanillaOption(
            option_type='call',
            underlying=self.underlying,
            strike=self.call_strike,
            ttm=self.ttm,
            pricing_model=self.pricing_model,
            quantity=self.quantity * self.multiplier,
            time_slice=self.time_slice)
        return call_option

    def _build_future(self):
        future = EquityFuture(
            underlying=self.underlying,
            ttm=self.ttm,
            quantity=(self.quantity * -1),
            time_slice=self.time_slice,
            multiplier=self.multiplier)
        return future

    @property
    def mtm(self):
        return self.future.mtm

    @mtm.setter
    def mtm(self, value):
        self.future.mtm = value

    @property
    def time_slice(self) -> TimeSlice:
        return self._time_slice

    @time_slice.setter
    def time_slice(self, sliceobj: TimeSlice) -> None:
        self._time_slice = sliceobj
        self.ttm -= sliceobj['time_delta']
        self.call.time_slice = self._time_slice
        self.future.time_slice = self._time_slice

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        self._quantity = value
        self.call.quantity = self.quantity * self.multiplier
        self.future.quantity = (self.quantity * -1)

    @property
    def value(self) -> float:
        return sum([self.call.value, self.future.value])

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'underlying={repr(self.underlying)}, '
            f'call_strike={repr(self.call_strike)}, '
            f'ttm={repr(self.ttm)}, '
            f'pricing_model={repr(self.pricing_model)}, '
            f'quantity={repr(self.quantity)}, '
            f'multiplier={repr(self.multiplier)}, '
            f'id={repr(self.id)})')
