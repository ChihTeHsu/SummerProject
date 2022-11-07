#from IPython import get_ipython
#get_ipython().magic('reset -f')
#get_ipython().run_line_magic('reset', '-f')
import seaborn as sns
import matplotlib.pyplot as plt

#sns.histplot(data, x="neighbourhood_group")

import numpy as np
import pandas as pd
data = pd.read_csv('accepted_2007_to_2018Q4.csv',parse_dates=['issue_d'], infer_datetime_format=True)
data_feat = data.columns.values

#%%
browse_notes = pd.read_excel('LCDataDictionary.xlsx',sheet_name=1)
browse_feat = browse_notes['BrowseNotesFile'].dropna().values
avail_feat = np.intersect1d(browse_feat, data_feat)

#%%
feat_dictionary = pd.read_csv('data dictionary.csv')
feat = feat_dictionary['Var'].dropna().values

#%%

select_feat = [
'acc_now_delinq',
'acc_open_past_24mths',
'all_util',
'annual_inc',
'avg_cur_bal',
'bc_open_to_buy',
'bc_util',
'chargeoff_within_12_mths',
'collections_12_mths_ex_med',
'delinq_2yrs',
'delinq_amnt',
'dti',
'earliest_cr_line',

'fico_range_high',
'fico_range_low',
'grade',

'il_util',
'inq_fi',
'inq_last_12m',
'inq_last_6mths',
'installment',
'int_rate',
'last_fico_range_high',
'last_fico_range_low',

'max_bal_bc',
'mo_sin_old_il_acct',
'mo_sin_old_rev_tl_op',
'mo_sin_rcnt_rev_tl_op',
'mo_sin_rcnt_tl',
'mort_acc',
'mths_since_last_delinq',
'mths_since_last_major_derog',
'mths_since_last_record',
'mths_since_rcnt_il',
'mths_since_recent_bc',
'mths_since_recent_bc_dlq', # same as mths_since_recent_loan_delinq
'mths_since_recent_inq',
'mths_since_recent_revol_delinq',
'num_accts_ever_120_pd',
'num_actv_bc_tl',
'num_actv_rev_tl',
'num_bc_sats',
'num_bc_tl',
'num_il_tl',
'num_op_rev_tl',
'num_rev_accts',
'num_rev_tl_bal_gt_0',
'num_sats',
'num_tl_120dpd_2m',
'num_tl_30dpd',
'num_tl_90g_dpd_24m',
'num_tl_op_past_12m',
'open_acc',
'open_acc_6m',
'open_il_12m',
'open_il_24m',
'open_rv_12m',
'open_rv_24m',

'pct_tl_nvr_dlq',
'percent_bc_gt_75',
'pub_rec',
'pub_rec_bankruptcies',

'revol_util',
'sub_grade',
'term',
'tot_coll_amt',
'tot_cur_bal',
'tot_hi_cred_lim',
'total_acc',
'total_bal_ex_mort',
'total_bal_il',
'total_bc_limit',
'total_cu_tl',
'total_il_high_credit_limit',
'total_rev_hi_lim',
'verification_status',]

X = data[select_feat].copy()
X.info()

#%%
# The features earliest_cr_line is date and its type should be changed to datetime. Later 
# it need to be transformed to ordinal numeric features
X['earliest_cr_line'] = pd.to_datetime(X['earliest_cr_line'], infer_datetime_format=True)
X['earliest_cr_line'].hist(bins=50)

#%%
# Below is the table of columns with missing values and their ratio to the total number of rows.
nan_mean = X.isna().mean()
nan_mean = nan_mean[nan_mean != 0].sort_values()
print(nan_mean)

#%%
print(data['loan_status'].value_counts())
y = data['loan_status'].copy()
y = y.isin(['Current', 'Fully Paid', 'In Grace Period']).astype('int')
y.value_counts()