#!/usr/bin/env python3

"""
BLACK SCHOLES MERTON OPTION PRICING MODEL (1973)

This module implements option pricing models and greek calculations.


The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

import math


def std_norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))


def std_norm_pdf(x):
    return (1 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * x ** 2)


def bsm_pricer(put_call: str, spot: float, strike: float, ttm: float,
               rfr: float, div_yield: float, sigma: float) -> float:
    """
    Implementation of the generalized Black Scholes Merton option
    pricing formula using continuous dividend yield.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: Spot price of the underlying
    :param strike: Strike price of the option
    :param ttm: Time to maturity (in year fractions)
    :param rfr: The continuously compounded annualized risk free rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: The implied volatility of the option
    :return: Option price
    """

    d1 = ((math.log(spot / strike)
          + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    d2 = d1 - sigma * math.sqrt(ttm)

    if put_call == 'call':
        price = (spot * math.exp(-div_yield * ttm) * std_norm_cdf(d1)
                 - strike * math.exp(-rfr * ttm) * std_norm_cdf(d2))

    elif put_call == 'put':
        price = (strike * math.exp(-rfr * ttm) * std_norm_cdf(-d2)
                 - spot * math.exp(-div_yield * ttm) * std_norm_cdf(-d1))

    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return price


def bsm_delta(put_call: str, spot: float, strike: float, ttm: float,
              rfr: float, div_yield: float, sigma: float) -> float:
    """
    Return Black-Scholes delta of an option.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: underlying asset price
    :param strike: strike price
    :param ttm: time to expiration in years
    :param rfr: risk-free interest rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: annualized standard deviation, or volatility
    :return: float of the BSM option delta
    """

    d1 = ((math.log(spot / strike)
          + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    if put_call == 'put':
        delta = math.exp(-div_yield * ttm) * std_norm_cdf(-d1) * - 1

    elif put_call == 'call':
        delta = math.exp(-div_yield * ttm) * std_norm_cdf(d1)

    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return delta


def bsm_gamma(put_call: str, spot: float, strike: float, ttm: float,
              rfr: float, div_yield: float, sigma: float) -> float:
    """
    Return Black-Scholes gamma of an option.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: underlying asset price
    :param strike: strike price
    :param ttm: time to expiration in years
    :param rfr: risk-free interest rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: annualized standard deviation, or volatility
    :return: float of the BSM option gamma
    """

    d1 = ((math.log(spot / strike)
          + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    gamma = (math.exp(-div_yield * ttm) * std_norm_pdf(d1)
             / (spot * sigma * math.sqrt(ttm)))

    if put_call not in ['put', 'call']:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return gamma


def bsm_theta(put_call: str, spot: float, strike: float, ttm: float,
              rfr: float, div_yield: float, sigma: float) -> float:
    """
    Return Black-Scholes theta of an option.

    NOTE: The traditional analytical solution is not expressed in days,
    however, it is in practice, thus the analytical theta is divided by 365.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: underlying asset price
    :param strike: strike price
    :param ttm: time to expiration in years
    :param rfr: risk-free interest rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: annualized standard deviation, or volatility
    :return: float of the BSM option theta
    """

    d1 = ((math.log(spot / strike)
          + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    d2 = d1 - sigma * math.sqrt(ttm)

    first_term = ((spot
                   * math.exp(-div_yield * ttm)
                   * std_norm_pdf(d1) * sigma)
                  / (2 * math.sqrt(ttm)))

    if put_call == 'call':

        second_term = (-div_yield * spot * math.exp(-div_yield * ttm)
                       * std_norm_cdf(d1))

        third_term = rfr * strike * math.exp(-rfr * ttm) * std_norm_cdf(d2)
        theta = -1 * (first_term + second_term + third_term) / 365.0

    elif put_call == 'put':

        second_term = (-div_yield * spot * math.exp(-div_yield * ttm)
                       * std_norm_cdf(-d1))

        third_term = rfr * strike * math.exp(-rfr * ttm) * std_norm_cdf(-d2)
        theta = (-first_term + second_term + third_term) / 365.0

    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return theta


def bsm_vega(put_call: str, spot: float, strike: float, ttm: float,
             rfr: float, div_yield: float, sigma: float) -> float:
    """
    Return Black-Scholes vega of an option.

    NOTE: The traditional analytical solution is not expressed in terms of
    1% IV changes like it is in practice, thus the vega is multiplied by 0.01.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: underlying asset price
    :param strike: strike price
    :param ttm: time to expiration in years
    :param rfr: risk-free interest rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: annualized standard deviation, or volatility
    :return: float of the BSM option vega
    """

    d1 = ((math.log(spot / strike)
           + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    vega = (spot * math.exp(-div_yield * ttm)
            * std_norm_pdf(d1) * math.sqrt(ttm))

    vega = vega * 0.01

    if put_call not in ['put', 'call']:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    return vega


def bsm_rho(put_call: str, spot: float, strike: float, ttm: float,
            rfr: float, div_yield: float, sigma: float) -> float:
    """
    Return Black-Scholes rho of an option.

    NOTE: The traditional analytical solution is not expressed in terms of
    1% rate changes like it is in practice, thus the rho is multiplied by 0.01.

    :param put_call: string of 'put' for put or 'call' for call
    :param spot: underlying asset price
    :param strike: strike price
    :param ttm: time to expiration in years
    :param rfr: risk-free interest rate
    :param div_yield: The continuously compounded annualized dividend yield
    :param sigma: annualized standard deviation, or volatility
    :return: float of the BSM option rho
    """

    d1 = ((math.log(spot / strike)
           + (rfr - div_yield + 0.5 * sigma ** 2) * ttm)
          / (sigma * math.sqrt(ttm)))

    d2 = d1 - sigma * math.sqrt(ttm)

    if put_call == 'call':
        rho = ttm * strike * math.exp(-rfr * ttm) * std_norm_cdf(d2)

    elif put_call == 'put':
        rho = -ttm * strike * math.exp(-rfr * ttm) * std_norm_cdf(-d2)

    else:
        raise ValueError("ERROR: put_call must be 'put' or 'call'")

    rho = rho * 0.01

    return rho


if __name__ == '__main__':

    # Validated against https://goodcalculators.com/black-scholes-calculator/

    params = {
        'put_call': 'call',
        'spot': 3200,
        'strike': 3200,
        'ttm': 0.25,
        'rfr': 0.02,
        'div_yield': 0.01,
        'sigma': 0.16,
    }

    price = bsm_pricer(**params)
    delta = bsm_delta(**params)
    gamma = bsm_gamma(**params)
    vega = bsm_vega(**params)
    theta = bsm_theta(**params)
    rho = bsm_rho(**params)

    print(f'Price: {round(price, 4)}')
    print(f'Delta: {round(delta, 4)}')
    print(f'Gamma: {round(gamma, 4)}')
    print(f'Vega: {round(vega, 4)}')
    print(f'Theta: {round(theta, 4)}')
    print(f'Rho: {round(rho, 4)}')
