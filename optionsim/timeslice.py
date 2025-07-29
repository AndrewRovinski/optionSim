from dataclasses import dataclass
from fractions import Fraction
from .volatilitysurface import VolatilitySurface


# ToDo: find clean way to incorporate multiple equity indices
@dataclass
class TimeSlice(object):
    """DataClass that represents a snapshot of market data"""
    time_step: int
    time_delta: Fraction
    spx_level: float
    dividend_yield: float
    interest_rates: float
    periodic_yield: float
    implied_volatility: VolatilitySurface

    def __getitem__(self, key):
        return getattr(self, key)
