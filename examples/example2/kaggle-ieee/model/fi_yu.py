import os

import numpy as np
import pandas as pd
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')


all_count = []
def _D2_series(DT, D2):
    len_       = len(DT)
    all_series = []
    series  = np.zeros(len_) 
    flag   = 0
    flagD  = 0
    for i in range(len_):
        if series[i]==0:
            flag += 1
            series[i] = flag  
        
        dt_1         = DT[i]
        d2_1         = D2[i] #+ D[i]
          
        for j in range(i+1, len_):
            dt_2         = DT[j]
            d2_2         = D2[j] 
              
            h_diff       = (dt_2 - dt_1) / 3600 / 24 # 
            d2_diff      =  d2_2 - d2_1              # 距离第一次的时间差 
            cond        = abs(h_diff - d2_diff) <= 0.99 # 相对时间差为 
#             if abs(h_diff) > 14 and cond == 1:
#                 print(h_diff,d2_2,d2_1)
#                 break
            if cond == 1: 
                series[j] =  series[i]
                break               
              
    return  series

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)+'_'+train_transaction['card5'].astype(str)+'_'+train_transaction['card6'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)+'_'+test_transaction['card5'].astype(str)+'_'+test_transaction['card6'].astype(str)

train_transaction['card123456_add1'] = train_transaction['uid'].astype(str)+'_'+train_transaction['addr1'].astype(str)
test_transaction['card123456_add1'] = test_transaction['uid'].astype(str)+'_'+test_transaction['addr1'].astype(str)

train_transaction = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

transaction_identity = train_transaction.append(test_transaction)

# transaction_identity['D3'] = transaction_identity['D3'] - transaction_identity['TransactionDT'].map(lambda x:(x//(3600*24)))
### sort
transaction_identity = transaction_identity.sort_values(['card123456_add1', 'D2','TransactionDT'])
D2        = transaction_identity.groupby('card123456_add1').apply(lambda x: _D2_series(x['TransactionDT'].values, x['D2'].values))
D2_series = []
for val in D2.values:
    D2_series.extend(val)

card_magic_fea_cols1_1 = []

transaction_identity['D2_series']                 = D2_series 
transaction_identity['card123456_add1_D2_series'] = transaction_identity['card123456_add1'].astype(str) + '_' + transaction_identity['D2_series'].astype(str)

transaction_identity = transaction_identity.sort_values(['card123456_add1', 'D15','TransactionDT'])
D15        = transaction_identity.groupby('card123456_add1').apply(lambda x: _D2_series(x['TransactionDT'].values, x['D15'].values))
D15_series = []
for val in D15.values:
    D15_series.extend(val)
transaction_identity['D15_series']                 = D15_series 
transaction_identity['card123456_add1_D15_series'] = transaction_identity['card123456_add1'].astype(str) + '_' + transaction_identity['D15_series'].astype(str)

transaction_identity = transaction_identity.sort_values(['card123456_add1', 'D11','TransactionDT'])
D11        = transaction_identity.groupby('card123456_add1').apply(lambda x: _D2_series(x['TransactionDT'].values, x['D11'].values))
D11_series = []
for val in D11.values:
    D11_series.extend(val)
transaction_identity['D11_series']                 = D11_series 
transaction_identity['card123456_add1_D11_series'] = transaction_identity['card123456_add1'].astype(str) + '_' + transaction_identity['D11_series'].astype(str)

transaction_identity = transaction_identity.sort_values(['card123456_add1', 'D8','TransactionDT'])
D8        = transaction_identity.groupby('card123456_add1').apply(lambda x: _D2_series(x['TransactionDT'].values, x['D8'].values))
D8_series = []
for val in D8.values:
    D8_series.extend(val)
transaction_identity['D8_series']                 = D8_series 
transaction_identity['card123456_add1_D8_series'] = transaction_identity['card123456_add1'].astype(str) + '_' + transaction_identity['D8_series'].astype(str)

transaction_identity['card123456_add1_D2_series_cnt'] = transaction_identity['card123456_add1_D2_series'].map(transaction_identity['card123456_add1_D2_series'].value_counts())

transaction_identity['card123456_D2_series_D3_mean']      = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('mean').values
transaction_identity['card123456_D2_series_D3_std']       = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('std').values
 
transaction_identity['card123456_D2_series_D3_skew']      = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('skew').values
transaction_identity['card123456_D2_series_D3_median']    = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('median').values
transaction_identity['card123456_D2_series_D3_max']       = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('max').values

transaction_identity['card123456_D2_series_D3_min']       = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('min').values
transaction_identity['card123456_D2_series_D3_quantile']  = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform('quantile').values
transaction_identity['card123456_D2_series_D3_kurtosis']  = transaction_identity.groupby('card123456_add1_D2_series')['D3'].transform(pd.DataFrame.kurtosis).values  

# transaction_identity['card123456_D2_series_D15_mean']      = transaction_identity.groupby('card123456_add1_D2_series')['D15'].transform('mean').values
# transaction_identity['card123456_D2_series_D15_std']       = transaction_identity.groupby('card123456_add1_D2_series')['D15'].transform('std').values
 
# transaction_identity['card123456_D2_series_D15_skew']      = transaction_identity.groupby('card123456_add1_D2_series')['D15'].transform('skew').values
# transaction_identity['card123456_D2_series_D15_median']    = transaction_identity.groupby('card123456_add1_D2_series')['D15'].transform('median').values
# transaction_identity['card123456_D2_series_D15_max']       = transaction_identity.groupby('card123456_add1_D2_series')['D15'].transform('max').values

# transaction_identity['card123456_D2_series_D15_min']       = transaction_identity.groupby('card123456_add1_D2_series')['D15'].transform('min').values

cols = ['card123456_add1_D2_series','card123456_add1_D2_series_cnt','card123456_D2_series_D3_mean','card123456_D2_series_D3_std','card123456_D2_series_D3_skew','card123456_D2_series_D3_median','card123456_D2_series_D3_max','card123456_D2_series_D3_min','card123456_D2_series_D3_quantile','card123456_D2_series_D3_kurtosis']

# ,'card123456_D2_series_D15_mean','card123456_D2_series_D15_std','card123456_D2_series_D15_skew','card123456_D2_series_D15_median','card123456_D2_series_D15_max','card123456_D2_series_D15_min'




cols = ['card123456_add1_D2_series','card123456_add1_D2_series_cnt','card123456_D2_series_D3_mean','card123456_D2_series_D3_std','card123456_D2_series_D3_skew','card123456_D2_series_D3_median','card123456_D2_series_D3_max','card123456_D2_series_D3_min','card123456_D2_series_D3_quantile','card123456_D2_series_D3_kurtosis','card123456_add1_D15_series','card123456_add1_D11_series','card123456_add1_D8_series']
train_transaction[cols] = transaction_identity.loc[train_transaction.index,cols]
test_transaction[cols] = transaction_identity.loc[test_transaction.index,cols]

train_transaction[cols].to_csv('../input/fi_train4.csv',header = True)
test_transaction[cols].to_csv('../input/fi_test4.csv',header = True)
