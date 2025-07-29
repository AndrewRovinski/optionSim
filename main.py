
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from fractions import Fraction

from optionsim.scenario import Scenario
from optionsim.portfolio import Portfolio
from optionsim.pricingmodels import bsm_pricer
from optionsim.strategies import NoStrategy, ShortPutStrategy,\
    BuyHoldFutures, BuyHoldRiskReversal
from optionsim.simulator import Simulator
from optionsim.economicgenerators import GeometricBrownianMotionGenerator, \
    FlatDividendYieldGenerator, FlatYieldCurveGenerator, \
    FlatVolatilitySurfaceGenerator

strategy = 'B&H risk reversal'
log_name = 'B&H 3M 3% Risk Reversal 7%drift 17.75%RV medianIV'

# Set file paths for CSV scenario data to be read from
scenario_files = {
    'spx_level': './scenarioCSV/equity_7d_1775RV.csv',
    'dividend_yield': './scenarioCSV/dividends.csv',
    'interest_rates': './scenarioCSV/rates.csv',
    'implied_volatility': './scenarioCSV/volsurface_hist.csv'}

# Set file paths for CSV scenario data to be saved to
file_map = {
    'spx_level': './scenarioCSV/equity_7d_1775RV.csv',
    'dividend_yield': './scenarioCSV/dividends.csv',
    'interest_rates': './scenarioCSV/rates.csv'}

multiprocess = False
use_scenario_files = True


if __name__ == '__main__':

    print('Staring simulation.')

    # Generate Scenarios
    print('Setting up the economic scenarios.')

    # Set arrays for generating the volatility surface
    pct_strikes = np.linspace(0.1, 2, 100)
    ttm = np.array([0, 0.25, 0.5, 0.75, 1])

    equity_gen = GeometricBrownianMotionGenerator(0.01, 0.1775, 'spx', 3200)
    div_gen = FlatDividendYieldGenerator(0.01)
    rate_gen = FlatYieldCurveGenerator(0.02)
    vol_gen = FlatVolatilitySurfaceGenerator(ttm, pct_strikes, 0.1775)

    scenario = Scenario(
        proj_length=10,
        time_delta=Fraction(1, 52),
        n_paths=1000,
        equity_gen=equity_gen,
        div_gen=div_gen,
        rate_gen=rate_gen,
        vol_gen=vol_gen)

    if use_scenario_files:
        scenario.from_files(scenario_files)
    else:
        scenario.generate(save_paths=True, file_map=file_map)

    print('Running the scenarios.')
    # Select the hedge strategy

    if strategy == 'none':

        trading_strategy = NoStrategy()

    elif strategy == 'short puts':

        trading_strategy = ShortPutStrategy(
            underlying='spx',
            pricing_model=bsm_pricer,
            percent_strike=1.0,
            ttm=Fraction(3, 12),
            max_positions=1)

    elif strategy == 'B&H futures':

        trading_strategy = BuyHoldFutures(
            underlying='spx',
            ttm=Fraction(3, 12),
            max_positions=1)

    elif strategy == 'B&H risk reversal':

        trading_strategy = BuyHoldRiskReversal(
            underlying='spx',
            pricing_model=bsm_pricer,
            otm_percent=0.03,
            ttm=Fraction(3, 12),
            max_positions=50)

    else:
        raise NotImplementedError

    # Add the Liability to the portfolio
    portfolio = Portfolio(
        trading_strategy=trading_strategy,
        borrow_costs=True,
        starting_capital=100000)

    # Run Simulation
    simulation = Simulator(
        portfolio=portfolio,
        deep_logging=False,
        log_greeks=False)

    if multiprocess:
        pool = mp.Pool(mp.cpu_count())

        # Create Proxy manager object instance
        manager = mp.Manager()

        # Load scenarios into Proxy Dict
        shared_paths = manager.dict(scenario.paths)

        # Map the scenario runs to the process pool
        results = [pool.apply_async(simulation.run, (path_num, path))
                   for path_num, path in shared_paths.items()]

        # Run the simulations and get the results
        simulation_results = [result.get() for result in tqdm(results)]

        # Close the pool, rejoin processes
        pool.close()
        pool.join()

    else:

        simulation_results = []
        for path_num, path in tqdm(scenario.paths.items()):
            temp_sim = copy.deepcopy(simulation)
            simulation_results.append(temp_sim.run(path_num, path))

    # Concatenate the log results, save to CSV
    simulation_results = pd.concat(simulation_results, sort=True)
    simulation_results.to_csv(f'./logs/{log_name}.csv')

    print('Simulation complete!')
