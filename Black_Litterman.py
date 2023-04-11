import pandas as pd
import numpy as np
from numpy.linalg import inv


# Q: views (Kx1 vector)
# P: projection or picks (KxN matrix)
# omega: covariance matrix representing the uncertainty of views


def implied_returns(delta, sigma: pd.DataFrame, w) -> pd.Series:
    """
    Compute implied returns based on CAPM: pi = delta * sigma * w
    """
    return delta * sigma.dot(w)


def proportional_prior(sigma, tau, p, diag=True) -> pd.DataFrame:
    """
    Compute simplified omega (assumes that omega is proportional to the variance of the prior)
    omega = diag( P*(tau*sigma)*P^T )
    """
    omega = p.dot(tau * sigma).dot(p.T)

    if diag:
        # Make a diag matrix from the diag elements of omega
        return pd.DataFrame(np.diag(np.diag(omega.values)), index=p.index, columns=p.index)

    return pd.DataFrame(omega, index=p.index, columns=p.index)


def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=0.02):
    """
    Computes the posterior mean and variance
    """
    # omega is a K*K matrix of the covariance of the views.
    # omega is diagonal as the views are required to be independent and uncorrelated.
    # inv(omega) is known as the confidence in the investor's views.
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)

    # N = w_prior.shape[0]  # How many assets
    # K = q.shape[0]  # How many views

    # Reverse-engineer to get implied returns
    pi = implied_returns(delta, sigma_prior, w_prior)

    # Adjust (scale) sigma by uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior

    # Use version that does not require the inverse of omega
    # mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    # sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(
    #     p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(
        inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(
        inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)

    return mu_bl, sigma_bl


def calc_risk_aversion(market_returns: pd.DataFrame, sharpe_ratio=None, est_return=None, rf=0.0) -> float:
    """
    Calculate estimated risk aversion using three different approaches
    (1) when estimated return is given
    (2) when Sharpe ratio is given
    (3) otherwise, estimate from MV formula
    """
    excess_return = market_returns - rf

    # if estimated expected return is given
    if est_return is not None:
        return est_return / np.var(market_returns.values)
    # if sharpe ratio is given
    if sharpe_ratio is not None:
        # delta = SR / (std of market)
        return sharpe_ratio / np.std(market_returns.values)
    # otherwise, delta = (expected excess return of market) / (variance of market)
    return np.mean(excess_return.values) / np.var(market_returns.values)