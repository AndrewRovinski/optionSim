#!/usr/bin/env python3
"""
BASE CLASS FOR RUNNING PORTFOLIO SIMULATION

"""

import pandas as pd
from dataclasses import asdict

from .portfolio import Portfolio


class Simulator(object):
    """Class for running the simulations."""

    def __init__(self, portfolio: Portfolio, deep_logging: bool = False,
                 log_greeks: bool = False) -> None:
        """
        :param portfolio: instance of the Portfolio object
        :param deep_logging: boolean for adding asset objects to logfiles
        :param log_greeks: boolean flag for adding option greeks to logs
        """

        self.portfolio = portfolio
        self.deep_logging = deep_logging
        self.log_greeks = log_greeks
        self.log = {}

    def run(self, path_num: int, path: dict) -> pd.DataFrame:
        """
        Main method for running the scenarios

        :param path_num: int of the path number to run
        :param path: dict of Timeslice objects
        :return: pd.DataFFrame of the logfile results
        """

        for time_slice in path:

            # Reset portfolio cash flows
            self.portfolio.reset_cashflow()

            # Generate interest on cash in the portfolio
            self.portfolio.earn_interest(time_slice)

            # Calculate asset and liabilities values
            self.portfolio.update_values(time_slice)

            # Run hedging strategy on portfolio
            self.portfolio.trading_strategy.generate_signals(
                self.portfolio, time_slice)

            # Log snapshot of portfolio
            self.record_data(t=time_slice['time_step'])

        log = pd.DataFrame(self.log).T
        slices = [asdict(ts) for ts in path]
        slices = pd.DataFrame(slices)
        slices['time_delta'] = slices['time_delta'].astype(float)
        slices.drop('implied_volatility', axis=1, inplace=True)
        log = pd.concat([log, slices], axis=1)
        log['Scenario Number'] = path_num
        log = log.reset_index(drop=True).set_index(
            ['Scenario Number', 'time_step'])

        return log

    def record_data(self, t: int) -> None:
        """
        Records a snapshot of the current scenario values

        :param t: int of the time step number
        :return: None
        """

        total_assets = self.portfolio.asset_value + self.portfolio.cash
        if t > 0:
            pnl = total_assets - self.log[0]['Total Assets']
        else:
            pnl = 0

        log_vars = {
            'Cash Balance': self.portfolio.cash,
            'Hedge Asset Value': self.portfolio.asset_value,
            'Total Assets': total_assets,
            'PnL': pnl,
            'Interest Earned': self.portfolio.interest,
            'Cash Flow from Mark to Market': self.portfolio.mtm_cash_flow,
            'Cash Flow from Expirations': self.portfolio.expiration_cash_flow,
            'Cash Flow from Spend': self.portfolio.spend}

        for asset, quantity in self.portfolio.trade_quantity.items():
            log_vars[f'{asset} qty'] = quantity

        for asset, quantity in self.portfolio.expiration_quantity.items():
            log_vars[f'{asset} expiration qty'] = quantity

        asset_types = set([asset.type for asset in self.portfolio.assets])

        for asset_type in asset_types:
            log_vars[f'{asset_type} qty'] = 0
            log_vars[f'{asset_type} value'] = 0

        for asset in self.portfolio.assets:
            log_vars[f'{asset.type} qty'] += asset.quantity
            log_vars[f'{asset.type} value'] = asset.value

        if self.deep_logging:
            log_vars['Portfolio Assets'] = [
                asset for asset in self.portfolio.assets]

        if self.log_greeks:
            pass

        self.log[t] = log_vars
