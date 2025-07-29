
import numpy as np

from .assets import Asset
from .strategies import Strategy
from .timeslice import TimeSlice


class Portfolio(object):
    """
    A base class representing a portfolio of
    positions (including both instruments and cash), determined
    on the basis of a set of signals provided by a Strategy.

    NOTE: Cash yield is currently assumed to follow a continuously
    compounded rate of interest, taken from the annual zero
    rate generated in the Scenario object. Further improvements
    will need to incorporate reconciliations/differentiations
    between discrete and continuous rates passed to the generators/objects.
    """

    def __init__(self, trading_strategy: Strategy,
                 borrow_costs: bool = False,
                 starting_capital: float = 0) -> None:
        """
        :param trading_strategy: an instance of the hedge strategy to use
        :param borrow_costs: boolean of whether to include cash borrow costs
        :param starting_capital: float of the (extra) cash to start with
        """

        self.trading_strategy = trading_strategy
        self.borrow_costs = borrow_costs
        self.assets = []
        self.cash = starting_capital
        self.interest = 0
        self.mtm_cash_flow = 0
        self.expiration_cash_flow = 0
        self.spend = 0
        self.expiration_quantity = {}
        self.trade_quantity = {}

    @property
    def asset_value(self) -> float:
        return sum([asset.value for asset in self.assets])

    def earn_interest(self, time_slice: TimeSlice) -> None:
        """
        Method to earn yield on the cash balance of the portfolio.

        :param time_slice: TimeSlice instance
        :return: None, adds or subtracts cash from the cash balance
        """

        self.interest = self.cash * (np.exp(time_slice['periodic_yield']) - 1)
        if self.borrow_costs:
            self.cash += self.interest
        else:
            if self.cash > 0:
                self.cash += self.interest

    def update_values(self, time_slice: TimeSlice) -> None:
        """
        Updates the portfolio values based on the current scenario slice,
        time step, pricing models, and shock to use for hedge exposure.

        NOTE: The current implementation does NOT update the liabilities
        implied volatility. Any changes made to the liabilities IV while
        mid-scenario will have to be separately implemented.

        :param time_slice: TimeSlice
        :return: None, assigns updated values to assets and portfolio object
        """

        for asset in tuple(self.assets):

            # Update asset time slices
            asset.time_slice = time_slice

            # Account for mark-to-market cash flows
            if hasattr(asset, 'mtm'):
                self.transact('mark_to_market', asset, asset.mtm)

            # Check if asset has matured/expired
            if asset.ttm == 0:
                self.transact('expire', asset, asset.value)

    def transact(self, transaction_type: str,
                 asset: Asset, value: float) -> None:
        """
        Method to conduct transactions in the asset portfolio. This method
        mainly exists to ensure proper logic in adding and removing
        positions, but also to allow for logging of specific cash flows.

        :param transaction_type: string of the transaction type
        :param asset: instance of a portfolio asset object
        :param value: float of the transaction value
        :return: None, adds or removes objects from the asset portfolio
        """

        if transaction_type == 'open':
            self.assets.append(asset)
            self.cash -= value
            self.spend -= value
            self.trade_quantity[asset.type] = asset.quantity

        elif transaction_type == 'close':
            self.assets.remove(asset)
            self.cash += value
            self.spend += value
            self.trade_quantity[asset.type] = asset.quantity

        elif transaction_type == 'expire':
            self.assets.remove(asset)
            self.cash += value
            self.expiration_cash_flow += value
            self.expiration_quantity[asset.type] = asset.quantity

        elif transaction_type == 'mark_to_market':
            self.cash += value
            self.mtm_cash_flow += value
            # Set asset MTM to 0 to prevent accidental double counting
            asset.mtm = 0

    def get_assets(self, asset_type) -> list:
        """
        Return a list of assets in the portfolio of a certain type

        :param asset_type: str of the asset type to filter
        :return: list of the asset objects that exist in the portfolio
        """

        asset_list = [asset for asset in self.assets
                      if isinstance(asset, asset_type)]

        return asset_list

    def reset_cashflow(self):
        """Resets the cash flow log amounts between time steps"""
        self.interest = 0
        self.mtm_cash_flow = 0
        self.expiration_cash_flow = 0
        self.spend = 0
        self.expiration_quantity = {}
        self.trade_quantity = {}

