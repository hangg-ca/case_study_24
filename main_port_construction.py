from data import data_handler
import cvxpy as cp
import numpy as np
import pandas as pd
from data import uti
from numpy.linalg import inv, pinv
from scipy.optimize import minimize

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
# leverage as parameter
leverage = cp.Parameter(nonneg=True)

# step 1: obtain risk parity portfolio as base portfolio
# your risk budget percent of total portfolio risk (equal risk)
x_t = [1 / sigma.shape[0]] * sigma.shape[0]
w0 = [1 / sigma.shape[0]] * sigma.shape[0]  # your initial guess is 1/n
cons = ({'type': 'eq', 'fun': uti.total_weight_constraint},
        {'type': 'ineq', 'fun': uti.long_only_constraint})
res = minimize(uti.risk_budget_objective, w0, args=[sigma, x_t], method='SLSQP', constraints=cons,
               options={'disp': True})
w_rb = res.x

# defining initial variables
x = cp.Variable(10)
ret = mu @ x - leverage * 0.04  # assume coupon rate of borrowing is 4%
risk = cp.quad_form(x, sigma)

# defining constraints
constraints = [cp.sum(x) == 1 + leverage,
               x >= 0,
               leverage <= 0.15,
               ret >= 0.08,
               # weight should be +/- 20% change of risk parity portfolio weight w_rb
               x <= w_rb + 0.10,
               x >= w_rb - 0.10
               ]
objective = cp.Minimize(risk)

# loop through leverage
leverage_values = [0.05, 0.1, 0.15]
risk_values = []
ret_values = []
std_values = []
# optimal portfolio
optimal_portfolio = pd.DataFrame(index=expected_return.index)
for lev in leverage_values:
    leverage.value = lev
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("status:", prob.status)
    risk_values.append(risk.value)
    std_values.append(risk.value ** 0.5)
    ret_values.append(ret.value)
    optimal_portfolio[lev] = x.value
    # calculate marginal risk contribution and risk contribution for each asset
    marginal_risk_contribution = sigma @ x.value
    risk_contribution = x.value * marginal_risk_contribution
    risk_contribution_pct = risk_contribution / risk_contribution.sum()

print('To be finished!')
