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

old_col = [x for x in train_transaction.columns]
old_col2 = [x for x in test_transaction.columns]
change_col = [x for x in train_transaction.columns if x not in ['TransactionDT','isFraud']]

res = pd.read_csv('./simple_offline20.csv', index_col='TransactionID')
train_transaction['pred'] = res.iloc[:,0]
train_transaction['error'] = abs(train_transaction['isFraud'] - train_transaction['pred'])

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)


col_magic = ["V127","V306","V308","V309",'V310',"V311","V312","V313","V314","V315","V316","V317","V318","V319","V320","V321"]
col_magic = ["V127",'V310']
train_transaction['V307'] = train_transaction['V307'].fillna(-1)
test_transaction['V307'] = test_transaction['V307'].fillna(-1)

for j in range(len(col_magic)):
    train_transaction[col_magic[j]] = train_transaction[col_magic[j]].fillna(-1).map(float)
    test_transaction[col_magic[j]] = test_transaction[col_magic[j]].fillna(-1).map(float)

cache = train_transaction[['uid','isFraud','ProductCD','TransactionAmt','addr1','TransactionDT','V307','error'] + col_magic].values
ukey_dict = {}
ukey2_dict = {}
ukey_V307_dict = {}
ukey_V306_dict = {}
ukey_V127_dict = {}
ukey_V308_dict = {}
ukey_V309_dict = {}
ukey_V310_dict = {}
ukey_V311_dict = {}
ukey_V312_dict = {}
ukey_V313_dict = {}
ukey_V314_dict = {}
ukey_V315_dict = {}
ukey_V316_dict = {}
ukey_V317_dict = {}
ukey_V318_dict = {}
ukey_V319_dict = {}
ukey_V320_dict = {}
ukey_V321_dict = {}


count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
l = []
l2 = []
for i in range(cache.shape[0]):
    uid = cache[i,0]
    isFraud = cache[i,1] 
    ProductCD = cache[i,2] 
    amt = (cache[i,3])
    addr1 = str(cache[i,4]) 
    V307 = float(cache[i,6]) 
    t = cache[i,5] // 3600
    err = (cache[i,7]) 
        
    
    
    ukey = uid + "_" + addr1
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[train_transaction.index[i]] = train_transaction.index[i]
    for j in range(len(col_magic)):
        eval("ukey_"+col_magic[j]+"_dict")[train_transaction.index[i]] = 0
    if len(ukey_dict[ukey]) >= 1:
        for j in range(min(15,len(ukey_dict[ukey]))):
            pos = -j - 1
            if abs(ukey_dict[ukey][pos][1] - t) > 3 * 24:
                continue
                
            if abs(ukey_dict[ukey][pos][4] - V307) > 0.05 or abs(ukey_dict[ukey][pos][4] - V307)/ukey_dict[ukey][pos][4] > 0.005:
                continue
                
            l.append(train_transaction.index[ukey_dict[ukey][pos][2]])
            l2.append(train_transaction.index[i])
            ukey2_dict[train_transaction.index[i]] = ukey2_dict[train_transaction.index[ukey_dict[ukey][pos][2]]]

            for j in range(len(col_magic)):
                if abs(ukey_dict[ukey][pos][6+j] - cache[i,8+j]) < 0.05:
                    eval("ukey_"+col_magic[j]+"_dict")[train_transaction.index[i]] = 1
                elif abs(ukey_dict[ukey][pos][6+j] + ukey_dict[ukey][pos][5] - cache[i,8+j]) < 0.05:
                    eval("ukey_"+col_magic[j]+"_dict")[train_transaction.index[i]] = 2

                
            if ukey_dict[ukey][pos][0] + isFraud >= 1:
                count1 += 1
            if ukey_dict[ukey][pos][0] != isFraud:
                print(ukey,train_transaction.index[ukey_dict[ukey][pos][2]],train_transaction.index[i],ukey_dict[ukey][pos][1],t,ukey_dict[ukey][pos][4],V307,abs(ukey_dict[ukey][pos][4] - V307)/ukey_dict[ukey][pos][4])
                count2 += 1
            if ukey_dict[ukey][pos][0] == 0 and isFraud == 0:
                if ukey_dict[ukey][pos][3] > err:
                    count3 += 1
                    count7 += ukey_dict[ukey][pos][3] - err
                if ukey_dict[ukey][pos][3] < err:
                    count4 += 1
                    count7 += ukey_dict[ukey][pos][3] - err
            if ukey_dict[ukey][pos][0] == 1 and isFraud == 1:
                if ukey_dict[ukey][pos][3] > err:
                    count5 += 1
                    count8 += ukey_dict[ukey][pos][3] - err
                if ukey_dict[ukey][pos][3] < err:
                    count6 += 1 
                    count8 += ukey_dict[ukey][pos][3] - err
            break
                    

    ukey_dict[ukey].append([isFraud,t,i,err,V307 + amt,amt] + cache[i,8:].tolist())
    
    if i % 50000 == 0:
        print(count1,count2,count3,count4,count5,count6,count7,count8)
        print("*")
    
print(count1,count2,count3,count4,count5,count6,count7,count8)    
# train_transaction.loc[l,change_col] = train_transaction.loc[l2,change_col].values 
# train_transaction.loc[l,change_col] = train_transaction.loc[l2,change_col].values 
# print(train_transaction.loc[2987079,change_col])
print("-")

cache = test_transaction[['uid','uid','ProductCD','TransactionAmt','addr1','TransactionDT','V307'] + col_magic].values
ukey_dict = {}
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
l = []
l2 = []
for i in range(cache.shape[0]):
    uid = cache[i,0]
    isFraud = cache[i,1] 
    ProductCD = cache[i,2] 
    amt = (cache[i,3])
    addr1 = str(cache[i,4]) 
    V307 = float(cache[i,6]) 
    t = cache[i,5] // 3600
    
    ukey = uid + "_" + addr1
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[test_transaction.index[i]] = test_transaction.index[i]
    for j in range(len(col_magic)):
        eval("ukey_"+col_magic[j]+"_dict")[test_transaction.index[i]] = 0
    if len(ukey_dict[ukey]) >= 1:
        for j in range(min(15,len(ukey_dict[ukey]))):
            pos = -j - 1
            if abs(ukey_dict[ukey][pos][1] - t) > 3 * 24:
                continue
                
            if abs(ukey_dict[ukey][pos][4] - V307) > 0.05 or abs(ukey_dict[ukey][pos][4] - V307)/ukey_dict[ukey][pos][4] > 0.005:
                continue
                
            l.append(test_transaction.index[ukey_dict[ukey][pos][2]])
            l2.append(test_transaction.index[i])
            ukey2_dict[test_transaction.index[i]] = ukey2_dict[test_transaction.index[ukey_dict[ukey][pos][2]]]

            for j in range(len(col_magic)):
                if abs(ukey_dict[ukey][pos][6+j] - cache[i,7+j]) < 0.05:
                    eval("ukey_"+col_magic[j]+"_dict")[test_transaction.index[i]] = 1
                elif abs(ukey_dict[ukey][pos][6+j] + ukey_dict[ukey][pos][5] - cache[i,7+j]) < 0.05:
                    eval("ukey_"+col_magic[j]+"_dict")[test_transaction.index[i]] = 2
            
            break
                    

    ukey_dict[ukey].append([isFraud,t,i,1,V307 + amt,amt] + cache[i,7:].tolist())
    if i % 50000 == 0:
        print("*")


train_transaction['ukey'] = train_transaction.index.map(ukey2_dict)    
test_transaction['ukey'] = test_transaction.index.map(ukey2_dict)      
for j in range(len(col_magic)):
    train_transaction[col_magic[j]+'_magic'] = train_transaction.index.map(eval("ukey_"+col_magic[j]+"_dict"))    
    test_transaction[col_magic[j]+'_magic'] = test_transaction.index.map(eval("ukey_"+col_magic[j]+"_dict"))   


print(train_transaction.groupby('ukey')['TransactionAmt'].count().value_counts().iloc[:10])
print(test_transaction.groupby('ukey')['TransactionAmt'].count().value_counts().iloc[:10])

print(train_transaction['V310_magic'].value_counts().iloc[:10])
print(test_transaction['V310_magic'].value_counts().iloc[:10])

train_transaction[['ukey']+list(map(lambda x:x+"_magic",col_magic))].to_csv('../input/fe_train3.csv',header = True)
test_transaction[['ukey']+list(map(lambda x:x+"_magic",col_magic))].to_csv('../input/fe_test3.csv',header = True)