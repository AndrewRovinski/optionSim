#!/usr/bin/env python3
"""
DYNAMIC HEDGING STRATEGY IMPLEMENTATIONS

"""
# ToDo: type hinting for this module

import pandas as pd
from typing import Callable
from fractions import Fraction
from abc import ABC, abstractmethod

from .assets import VanillaOption, EquityFuture


class Strategy(ABC):
    """Abstract Base Class for Strategies"""
    @abstractmethod
    def generate_signals(self, portfolio, time_slice):
        pass


class NoStrategy(Strategy):
    """A base class used as a placeholder for non-hedged simulations"""
    def __init__(self):
        super().__init__(shock_percent=0, target_exposure=0)

    def generate_signals(self, portfolio, time_slice):
        pass


class ShortPutStrategy(Strategy):

    def __init__(self, underlying: str, pricing_model: Callable,
                 percent_strike: float, ttm: Fraction,
                 max_positions: int) -> None:

        self.underlying = underlying
        self.pricing_model = pricing_model
        self.percent_strike = percent_strike
        self.ttm = ttm
        self.max_positions = max_positions

    def generate_signals(self, portfolio, time_slice):

        # Check for existing positions to modify
        for asset in portfolio.assets:

            if isinstance(asset, VanillaOption):
                break

        else:
            current_spot = time_slice[f'{self.underlying}_level']

            put = VanillaOption(
                option_type='put',
                underlying=self.underlying,
                strike=(current_spot * self.percent_strike),
                ttm=self.ttm,
                quantity=self.max_positions * -1,
                pricing_model=self.pricing_model,
                time_slice=time_slice)

            # Add assets to portfolio, subtract cash from portfolio
            portfolio.transact(
                transaction_type='open',
                asset=put,
                value=put.value)


class BuyHoldFutures(Strategy):

    def __init__(self, underlying: str, ttm: Fraction,
                 max_positions: int) -> None:

        self.underlying = underlying
        self.ttm = ttm
        self.max_positions = max_positions

    def generate_signals(self, portfolio, time_slice):

        # Check for existing positions to modify
        for asset in portfolio.assets:

            if isinstance(asset, EquityFuture):
                break

        else:

            asset = EquityFuture(
                underlying=self.underlying,
                ttm=self.ttm,
                quantity=self.max_positions,
                time_slice=time_slice)

            # Add assets to portfolio, subtract cash from portfolio
            portfolio.transact(
                transaction_type='open',
                asset=asset,
                value=asset.value)


class BuyHoldRiskReversal(Strategy):

    def __init__(self, underlying: str, pricing_model: Callable,
                 otm_percent: float, ttm: Fraction,
                 max_positions: int) -> None:

        self.underlying = underlying
        self.pricing_model = pricing_model
        self.otm_percent = otm_percent
        self.ttm = ttm
        self.max_positions = max_positions

    def generate_signals(self, portfolio, time_slice):

        # Check for existing positions to modify
        for asset in portfolio.assets:

            if isinstance(asset, VanillaOption):
                break

        else:

            current_spot = time_slice[f'{self.underlying}_level']

            put = VanillaOption(
                option_type='put',
                underlying=self.underlying,
                strike=(current_spot * (1 - self.otm_percent)),
                ttm=self.ttm,
                quantity=self.max_positions * -1,
                pricing_model=self.pricing_model,
                time_slice=time_slice)

            # Add assets to portfolio, subtract cash from portfolio
            portfolio.transact(
                transaction_type='open',
                asset=put,
                value=put.value)

            call = VanillaOption(
                option_type='call',
                underlying=self.underlying,
                strike=(current_spot * (1 + self.otm_percent)),
                ttm=self.ttm,
                quantity=self.max_positions,
                pricing_model=self.pricing_model,
                time_slice=time_slice)

            # Add assets to portfolio, subtract cash from portfolio
            portfolio.transact(
                transaction_type='open',
                asset=call,
                value=call.value)
