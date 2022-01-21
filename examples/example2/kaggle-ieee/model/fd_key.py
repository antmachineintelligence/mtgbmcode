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

train_transaction = train_transaction.merge(train_identity , how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_identity , how='left', left_index=True, right_index=True)


res = pd.read_csv('./simple_offline20.csv', index_col='TransactionID')
train_transaction['pred'] = res.iloc[:,0]
train_transaction['error'] = abs(train_transaction['isFraud'] - train_transaction['pred'])

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)



id_col = ["id_31","id_30","DeviceInfo"]
bad_word = ['MacOS','Windows 7','mobile safari 11.0','mobile safari generic','Windows 10',
            'chrome 63.0','iOS Device','Windows']
train_transaction[id_col] = train_transaction[id_col].astype(str)
test_transaction[id_col] = test_transaction[id_col].astype(str)
cache = train_transaction[['uid','isFraud','ProductCD','TransactionAmt','addr1','TransactionDT','error'] + id_col].values
ukey_dict = {}
ukey2_dict = {}


count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
id_score = {}
match_id_np = []
for i in range(cache.shape[0]):
    uid = cache[i,0]
    isFraud = cache[i,1] 
    ProductCD = cache[i,2] 
    amt = (cache[i,3])
    addr1 = str(cache[i,4]) 
    t = cache[i,5] // 3600
    err = (cache[i,6]) 
    match_id = ["0"] * len(id_col)    
    
    
    ukey = str(amt) + addr1
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[train_transaction.index[i]] = train_transaction.index[i]
    if len(ukey_dict[ukey]) >= 1:
        isfit = True
        for j in range(min(5,len(ukey_dict[ukey]))):
            pos = -j - 1
            if abs(ukey_dict[ukey][pos][1] - t) > 2:
                continue

            isfit = False
            for j in range(len(id_col)):
                if cache[i,7+j] != 'nan' and cache[i,7+j] == ukey_dict[ukey][pos][5+j]:
                    isfit = True
                    if True:
                        match_id[j] = (cache[i,7+j])
                if cache[i,7+j] != 'nan' and ukey_dict[ukey][pos][5+j] != 'nan' and cache[i,7+j] != ukey_dict[ukey][pos][5+j]:
                    isfit = False
                    match_id = ["0"] * len(id_col) 
                    break
#                     break
            if not isfit or len(list(filter(lambda x:x!="0",match_id)))<2:
                continue
            
#             if uid == ukey_dict[ukey][pos][4]:
#                 continue
            
            ukey2_dict[train_transaction.index[i]] = ukey2_dict[train_transaction.index[ukey_dict[ukey][pos][2]]]
            match_id_np[ukey_dict[ukey][pos][2]] = match_id
            for col in match_id:
                id_score[col] = id_score.get(col,[0,0,0])
                id_score[col][0] += 1  
            count3 += 1
            if ukey_dict[ukey][pos][0] + isFraud >= 1:
                count1 += 1
            if ukey_dict[ukey][pos][0] != isFraud:
                print(train_transaction.index[ukey_dict[ukey][pos][2]],train_transaction.index[i])
                count2 += 1
            else:
                if isFraud == 1:
                    for col in match_id:
                        id_score[col] = id_score.get(col,[0,0,0])
                        id_score[col][1] += 1       
                else:
                    for col in match_id:
                        id_score[col] = id_score.get(col,[0,0,0])
                        id_score[col][2] += 1 
            break
                    

    ukey_dict[ukey].append([isFraud,t,i,err,uid] + cache[i,7:].tolist())
    match_id_np.append(match_id)
    if i % 50000 == 0:
        print(count1,count2,count3,count4,count5,count6,count7,count8)
        print("*")
    
print(count1,count2,count3,count4,count5,count6,count7,count8)    
# train_transaction.loc[l,change_col] = train_transaction.loc[l2,change_col].values 
# train_transaction.loc[l,change_col] = train_transaction.loc[l2,change_col].values 
# print(train_transaction.loc[2987079,change_col])
print("-")
f = open('./debug.csv','w')
for k,v in sorted(id_score.items(),key = lambda x:x[1][1]):
    print(k,v,v[1]/v[0],file = f)
f.close()

cols = []    
for j in range(len(id_col)):
    cols.append(id_col[j]+"_raw")
    train_transaction[id_col[j]+"_raw"] = np.array(match_id_np)[:,j]
    cols.append(id_col[j]+"_keycount")
    train_transaction[id_col[j]+"_keycount"] = train_transaction[id_col[j]+"_raw"].map(lambda x:id_score.get(x,["0","0","0"])[0])
    
cache = test_transaction[['uid','uid','ProductCD','TransactionAmt','addr1','TransactionDT','P_emaildomain']+ id_col].values
ukey_dict = {}
id_score = {}
match_id_np = []
for i in range(cache.shape[0]):
    uid = cache[i,0]
    isFraud = cache[i,1] 
    ProductCD = cache[i,2] 
    amt = (cache[i,3])
    addr1 = str(cache[i,4]) 
    t = cache[i,5] // 3600
    match_id = ["0"] * len(id_col) 
    
    ukey = str(amt) + addr1
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[test_transaction.index[i]] = test_transaction.index[i]
    if len(ukey_dict[ukey]) >= 1:
        isfit = True
        for j in range(min(5,len(ukey_dict[ukey]))):
            pos = -j - 1
            if abs(ukey_dict[ukey][pos][1] - t) > 2:
                continue

            isfit = False
            for j in range(len(id_col)):
                if cache[i,7+j] != 'nan' and cache[i,7+j] == ukey_dict[ukey][pos][5+j]:
                    isfit = True
                    if True:
                        match_id[j] = (cache[i,7+j])
                if cache[i,7+j] != 'nan' and ukey_dict[ukey][pos][5+j] != 'nan' and cache[i,7+j] != ukey_dict[ukey][pos][5+j]:
                    isfit = False
                    match_id = ["0"] * len(id_col) 
                    break
            if not isfit or len(list(filter(lambda x:x!="0",match_id)))<2:
                continue
#             if uid == ukey_dict[ukey][pos][4]:
#                 continue            
            ukey2_dict[test_transaction.index[i]] = ukey2_dict[test_transaction.index[ukey_dict[ukey][pos][2]]]
            match_id_np[ukey_dict[ukey][pos][2]] = match_id
            for col in match_id:
                id_score[col] = id_score.get(col,[0,0,0])
                id_score[col][0] += 1             
            break
                    

    ukey_dict[ukey].append([isFraud,t,i,1,uid] + cache[i,7:].tolist() )
    match_id_np.append(match_id)
    if i % 50000 == 0:
        print("*")


cols = []    
for j in range(len(id_col)):
    cols.append(id_col[j]+"_raw")
    test_transaction[id_col[j]+"_raw"] = np.array(match_id_np)[:,j]
    cols.append(id_col[j]+"_keycount")
    test_transaction[id_col[j]+"_keycount"] = test_transaction[id_col[j]+"_raw"].map(lambda x:id_score.get(x,["0","0","0"])[0])
    
train_transaction['tempkey'] = train_transaction.index.map(ukey2_dict)    
test_transaction['tempkey'] = test_transaction.index.map(ukey2_dict)      


print(train_transaction.groupby('tempkey')['TransactionAmt'].count().value_counts().iloc[:10])
print(test_transaction.groupby('tempkey')['TransactionAmt'].count().value_counts().iloc[:10])
print(cols)
train_transaction[['tempkey'] + cols].to_csv('../input/fd_train4.csv',header = True)
test_transaction[['tempkey'] + cols].to_csv('../input/fd_test4.csv',header = True)