from data import data_handler
import cvxpy as cp

# read data from excel
expected_return_vol = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                               sheet_name='ExpectedReturn').set_index('Asset')
expected_corr = data_handler.DataHandler.load_excel_data('./data/capital_market_assumption.xlsx',
                                                         sheet_name='Corr').set_index('Asset')
expected_vol = expected_return_vol.loc[:, 'My Vol'].to_frame('ExpectedVol')
print('To be finished!')
