import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv, pinv
from scipy.optimize import minimize
from data import data_handler, uti


# risk budgeting optimization
def calculate_portfolio_var(w, V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w * V * w.T)[0, 0]


def calculate_risk_contribution(w, V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w, V))
    # Marginal Risk Contribution
    MRC = V * w.T
    # Risk Contribution
    RC = np.multiply(MRC, w.T) / sigma
    return RC


def risk_budget_objective(x, pars):
    # calculate portfolio risk
    V = pars[0]  # covariance table
    x_t = pars[1]  # risk target in percent of portfolio risk
    sig_p = np.sqrt(calculate_portfolio_var(x, V))  # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p, x_t))
    asset_RC = calculate_risk_contribution(x, V)
    J = sum(np.square(asset_RC - risk_target.T))[0, 0]  # sum of squared error
    return J


def total_weight_constraint(x):
    return np.sum(x) - 1.0


def long_only_constraint(x):
    return x


# read data from excel
expected_return_vol = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                               sheet_name='ExpectedReturn').set_index('Asset')
expected_corr = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                         sheet_name='Corr').set_index('Asset')
expected_vol = expected_return_vol.loc[:, 'My Vol'].to_frame('ExpectedVol')
expected_return = expected_return_vol.loc[:, 'My Return'].to_frame('ExpectedReturn')
# variance covariance matrix from vol and corr
vcov = expected_vol.dot(expected_vol.T) * expected_corr
if not uti.isPD(vcov):
    vcov = uti.nearestPD(vcov)
    corr_implied = np.diag(1 / np.sqrt(np.diag(vcov))) @ vcov @ np.diag(1 / np.sqrt(np.diag(vcov)))
    vol_implied = np.diag(vcov) ** 0.5

# set up the optimization problem
mu = expected_return.values.reshape(1, -1)
sigma = vcov.values

