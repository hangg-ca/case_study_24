from data import data_handler
import cvxpy as cp
import numpy as np
import pandas as pd
from data import uti
from numpy.linalg import inv, pinv
from scipy.optimize import minimize

########################################################################################################################
# This code is only for demonstration purpose and not for real trading. Any changes are not suggested by the creator of
# this code. The creator of this code is not responsible for any loss caused by using this code.
# The user should use this code at their own risk.
# Creator: Hangg https://github.com/hangg-ca
########################################################################################################################

# read data from excel
borrowing_rate = 0.04 # assume borrowing rate is 4% for the company in CAD currency and AAA rating

expected_return_vol = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                               sheet_name='ExpectedReturn').set_index('Asset')
expected_corr = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                         sheet_name='Corr').set_index('Asset')
expected_vol = expected_return_vol.loc[:, 'My Vol'].to_frame('ExpectedVol')
expected_return = expected_return_vol.loc[:, 'My Return'].to_frame('ExpectedReturn')

equity_idx = expected_return.index.isin(['US Equity', 'International Developed Equity',
                                         'Emerging Equity', 'Private Equity'])
fi_idx = expected_return.index.isin(['Govt', 'IG Corp', 'TIPS', 'HY Corp'])
real_asset_idx = expected_return.index.isin(['Real Estate', 'Infrastructure'])

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
# min return as parameter
min_return = cp.Parameter(nonneg=True)

################################################
# step 1: obtain risk parity portfolio as base portfolio
################################################
# your risk budget percent of total portfolio risk (equal risk)
x_t = [1 / sigma.shape[0]] * sigma.shape[0]
w0 = [1 / sigma.shape[0]] * sigma.shape[0]  # your initial guess is 1/n
cons = ({'type': 'eq', 'fun': uti.total_weight_constraint},
        {'type': 'ineq', 'fun': uti.long_only_constraint})
res = minimize(uti.risk_budget_objective, w0, args=[sigma, x_t], method='SLSQP', constraints=cons,
               options={'disp': True})
w_rb = res.x

################################################
# step 2: Use efficient frontier to find optimal portfolio with different leverage and min return
################################################
# defining initial variables
x = cp.Variable(10)
ret = mu @ x - leverage * borrowing_rate
risk = cp.quad_form(x, sigma)

# defining constraints
constraints = [cp.sum(x) == 1 + leverage,
               x >= 0,
               ret >= min_return,
               # weight should be +/- 10% of risk parity portfolio weight w_rb
               x <= w_rb + 0.10,
               x >= w_rb - 0.10,
               # # asset class level weight should be less than 50% of total weight
               cp.sum(x[equity_idx]) <= 0.5,
               cp.sum(x[fi_idx]) <= 0.5,
               cp.sum(x[real_asset_idx]) <= 0.5,
               ]
objective = cp.Minimize(risk)

# loop through leverage
leverage_values = [0, 0.05, 0.1, 0.15, 0.2]
min_return_values = np.linspace(0.02, 0.1, 50)
risk_values = []
ret_values = []
std_values = []
# optimal portfolio
optimal_portfolio = pd.DataFrame(index=expected_return.index)
risk_return = pd.DataFrame(columns=['return', 'risk', 'leverage'])
for lev in leverage_values:
    for min_ret in min_return_values:
        min_return.value = min_ret
        leverage.value = lev
        prob = cp.Problem(objective, constraints)
        prob.solve()
        print('Set up parameters: leverage =', lev, 'min_return =', min_ret)
        print("status:", prob.status)
        if prob.status == 'optimal':
            risk_values.append(risk.value)
            std_values.append(risk.value ** 0.5)
            ret_values.append(ret.value)
            col_name = 'Leverage_' + str(lev) + '_MinRet_' + str(min_ret)
            optimal_portfolio[col_name] = x.value
            risk_return.loc[col_name, 'return'] = ret.value
            risk_return.loc[col_name, 'risk'] = risk.value ** 0.5
            risk_return.loc[col_name, 'leverage'] = lev
            # calculate marginal risk contribution and risk contribution for each asset
            marginal_risk_contribution = sigma @ x.value
            risk_contribution = x.value * marginal_risk_contribution
            risk_contribution_pct = risk_contribution / risk_contribution.sum()

# plot efficient frontier line with different leverage
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for lev in leverage_values:
    # plot efficient frontier line
    plt.plot(risk_return.loc[risk_return['leverage'] == lev, 'risk'],
             risk_return.loc[risk_return['leverage'] == lev, 'return'], label='Leverage_' + str(lev))
    plt.xlabel('Standard Deviation')
    plt.ylabel('Return')
    plt.title('Efficient Frontier with Different Leverage')
    plt.legend()
# save plot
plt.savefig('./result/efficient_frontier.png')

# save multiple optimal portfolio
optimal_portfolio.to_excel('./result/optimal_portfolio_combination.xlsx')
risk_return.to_excel('./result/risk_return.xlsx')
################################################
# Step 3: Get the optimal portfolio with defined leverage and min return
################################################
min_return.value = 0.08
leverage.value = 0.15
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
optimal_portfolio = pd.DataFrame(index=expected_return.index)
optimal_portfolio['OptimalPortfolio'] = x.value
optimal_portfolio['Risk Parity Portfolio'] = w_rb
optimal_portfolio['Risk Contribution'] = uti.calculate_risk_contribution(x.value, sigma)
optimal_portfolio['Risk Parity Contribution'] = uti.calculate_risk_contribution(w_rb, sigma)

optimal_portfolio.loc['Cash', 'OptimalPortfolio'] = 1 - x.value.sum()
optimal_portfolio.to_excel('./result/min_vol_optimal_portfolio.xlsx')

portfolio_characteristics = pd.DataFrame(index=['Expected Return', 'Expected Risk'])
portfolio_characteristics.loc['Expected Return', 'OptimalPortfolio'] = ret.value
portfolio_characteristics.loc['Expected Risk', 'OptimalPortfolio'] = risk.value ** 0.5
portfolio_characteristics.to_excel('./result/min_vol_portfolio_characteristics.xlsx')

################################################
# Final Step: Tilt the portfolio to improve the risk contribution and draw scenario
################################################
# read the final portfolio
tilt_portfolio = data_handler.DataHandler.load_excel_data('optimal_portfolio_final.xlsx', sheet_name='Sheet1')
tilt_portfolio = tilt_portfolio.set_index('Asset')[['FinalPortfolio']]
tilt_portfolio['Risk Contribution'] = uti.calculate_risk_contribution(tilt_portfolio['FinalPortfolio'], sigma)
portfolio_characteristics = pd.DataFrame(index=['Expected Return', 'Expected Risk'])
portfolio_characteristics.loc['Expected Return', 'TiltPortfolio'] = (mu @ tilt_portfolio['FinalPortfolio'] -
                                                                     leverage.value * borrowing_rate)
portfolio_characteristics.loc['Expected Risk', 'TiltPortfolio'] = uti.calculate_portfolio_var(
    tilt_portfolio['FinalPortfolio'], sigma) ** 0.5
tilt_portfolio.to_excel('./result/optimal_portfolio_final.xlsx')
portfolio_characteristics.to_excel('./result/optimal_portfolio_characteristics.xlsx')

################################################
# Scenario analysis
################################################
scenarios_rst = pd.DataFrame()
scenarios_rst_port = pd.DataFrame(index=expected_return.index)
# 1. What if the bond equity correlation increases by 0.3? Scenario 1
# 2. What if the Real Estate correlation with Equity and Bond increases by 0.3? Scenario 2
for scenario_cases in ['Base Scenario', 'Corr_Scenario1', 'Corr_Scenario2']:
    if scenario_cases == 'Base Scenario':
        new_corr = expected_corr.copy()
    else:
        new_corr = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                            sheet_name=scenario_cases).set_index('Asset Class')
    # variance covariance matrix from vol and corr
    new_vcov = expected_vol.dot(expected_vol.T) * new_corr
    if not uti.isPD(new_vcov):
        new_vcov = uti.nearestPD(new_vcov)
        corr_implied = np.diag(1 / np.sqrt(np.diag(new_vcov))) @ new_vcov @ np.diag(1 / np.sqrt(np.diag(new_vcov)))
        vol_implied = np.diag(new_vcov) ** 0.5
    sigma = new_vcov.values
    scenarios_rst.loc[scenario_cases, 'final ptf std'] = uti.calculate_portfolio_var(
        tilt_portfolio['FinalPortfolio'].values, sigma) ** 0.5
    risk = cp.quad_form(x, sigma)
    # defining constraints
    constraints = [cp.sum(x) == 1 + leverage,
                   x >= 0,
                   ret >= min_return,
                   x <= w_rb + 0.10,
                   x >= w_rb - 0.10,
                   ]
    objective = cp.Minimize(risk)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("status:", prob.status)
    optimal_portfolio = pd.DataFrame(index=expected_return.index)
    scenarios_rst_port[scenario_cases + ' Wgt'] = x.value
    scenarios_rst_port[scenario_cases + ' Risk Parity Portfolio'] = uti.calculate_risk_contribution(x.value, sigma)
    scenarios_rst.loc[scenario_cases, 'rerun ptf std'] = uti.calculate_portfolio_var(x.value, sigma) ** 0.5

scenarios_rst.to_excel('./result/scenario_analysis.xlsx')
scenarios_rst_port.to_excel('./result/scenario_analysis_port.xlsx')
print('Done!')
