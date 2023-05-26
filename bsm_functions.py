# # Valuation of European call options in Black-Scholes-Merton model
#  incl. Vega function and implied volatility estimation
#  bsm_functions.py
#
#  Analytical Black-Scholes-Merton (BSM) Formula

from math import log, sqrt, exp
from scipy import stats

def bsm_call_value( S0, K, T, r, sigma):
    """ Valuation of European call option in BSM model.
    Hilpisch, Yves. Python for Finance: Analyze Big Financial Data (Kindle Locations 1471-1517).
    O'Reilly Media. Kindle Edition.

    Analytical formula.

    Parameters = = = = = = = = = =
     S0 : float initial stock/ index level
     K : float strike price
     T : float maturity date (in year fractions)
     r : float constant risk-free short rate
     sigma : float volatility factor in diffusion term

     Returns = = = = = = =
     value : float present value of the European call option """

    S0 = float(S0)
    d1 = (log( S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt( T))
    d2 = (log( S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt( T))
    value = (S0 * stats.norm.cdf( d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf( d2, 0.0, 1.0))

    # stats.norm.cdf --> cumulative distribution function
    # for normal distribution
    return value

# Vega Function

def bsm_vega( S0, K, T, r, sigma):
    """ Vega of European option in BSM model.
    Hilpisch, Yves. Python for Finance: Analyze Big Financial Data (Kindle Locations 1518-1551).
    O'Reilly Media. Kindle Edition.

    Parameters = = = = = = = = = =
    S0 : float initial stock/ index level
    K : float strike price
    T : float maturity date (in year fractions)
    r : float constant risk-free short rate
    sigma : float volatility factor in diffusion term

    Returns = = = = = = =
    vega : float partial derivative of BSM formula with respect to sigma, i.e. Vega """

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.cdf( d1, 0.0, 1.0) * sqrt(T)
    return vega


# Implied Volatility Function
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it = 100):
    """ Implied volatility of European call option in BSM model.
    Hilpisch, Yves. Python for Finance: Analyze Big Financial Data (Kindle Locations 1552-1580).
    O'Reilly Media. Kindle Edition.

    Parameters = = = = = = = = = =
    S0 : float initial stock/ index level
    K : float strike price
    T : float maturity date (in year fractions)
    r : float constant risk-free short rate
    sigma_est : float estimate of impl. volatility
    it : integer number of iterations

    Returns = = = = = = =
    simga_est : float numerically estimated implied volatility """

    for i in range(it):
        sigma_est -= (( bsm_call_value( S0, K, T, r, sigma_est) - C0) / bsm_vega( S0, K, T, r, sigma_est))

    return sigma_est


