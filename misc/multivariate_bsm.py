
import numpy as np
import pandas as pd

from optionsim.pricingmodels import bsm_pricer


def generate_paths(initial_price: float, asset1_sigma: float,
                   asset2_sigma: float, mu: float, rho: float,
                   proj_length: int, time_delta: float,
                   n_paths: int, rand_seed: int = None) -> pd.DataFrame:


    np.random.seed(rand_seed)

    n_steps = int(proj_length / time_delta)

    # Generate errors for asset 1
    b1 = np.random.normal(
        scale=asset1_sigma * np.sqrt(time_delta),
        size=(n_paths, n_steps))

    # Generate random errors for asset 2
    y = np.random.normal(
        scale=asset2_sigma * np.sqrt(time_delta),
        size=(n_paths, n_steps))

    # Generate correlated errors for asset 2
    b2 = rho * b1 + np.sqrt(1 - rho ** 2) * y

    # Use cumulative sum of errors to generate paths
    b1 = np.cumsum(b1, axis=1)
    b1 = np.insert(b1, 0, 0, axis=1)
    b2 = np.cumsum(b2, axis=1)
    b2 = np.insert(b2, 0, 0, axis=1)

    # Generate an array of time steps
    t = np.arange(0, proj_length + time_delta, time_delta)
    t = np.tile(t, (n_paths, 1))

    # Generate the drift terms
    d1 = t * (mu - 0.5 * asset1_sigma ** 2)
    d2 = t * (mu - 0.5 * asset2_sigma ** 2)

    # Multiply the initial price by the cumulative drift + errors
    s1 = initial_price * np.exp(d1 + b1)
    s2 = initial_price * np.exp(d2 + b2)

    # Calculate basket prices
    basket = 0.5 * s1 + 0.5 * s2

    # Create dataframe with the paths
    t_steps = pd.Series(np.arange(0, n_steps + 1))
    t_steps = 't_' + t_steps.astype(str)
    basket = pd.DataFrame(basket, columns=t_steps)
    basket.index = basket.index + 1

    return basket


def price_call_options(basket_df: pd.DataFrame,
                       mu: float, proj_length: int,
                       initial_price: float,
                       moneyness: list) -> pd.DataFrame:

    terminal_vals = basket_df.iloc[:, -1]

    strikes = [k * initial_price for k in moneyness]

    option_prices = pd.DataFrame(index=terminal_vals.index)

    for strike in strikes:
        option_vals = terminal_vals.apply(lambda x: max(0, x - strike))
        option_vals = option_vals * np.exp(-mu * proj_length)
        option_prices[f'{strike}_strike'] = option_vals

    option_prices = option_prices.T

    results = option_prices.mean(axis=1)
    results.name = 'Numerical mvBSM Price'
    results = results.to_frame()

    return results


if __name__ == '__main__':

    initial_price = 100
    asset1_sigma = 0.2
    asset2_sigma = 0.2
    mu = 0.01
    rho = 1
    proj_length = 1
    time_delta = 1 / 252
    n_paths = 10000
    moneyness = [0.75, 1.25]

    scenarios = generate_paths(
        initial_price, asset1_sigma, asset2_sigma,
        mu, rho, proj_length, time_delta, n_paths)

    results = price_call_options(scenarios, mu, proj_length,
                                 initial_price, moneyness)

    option_price1 = bsm_pricer(
        'call', initial_price, initial_price * moneyness[0],
        1, 0.01, 0, 0.2)


    option_price2 = bsm_pricer(
        'call', initial_price, initial_price * moneyness[1],
        1, 0.01, 0, 0.2)

    results['BSM Price'] = [option_price1, option_price2]

    print(results)
