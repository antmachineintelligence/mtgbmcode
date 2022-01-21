import os

import numpy as np
import pandas as pd
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)


train_test = train_transaction.append(test_transaction)
train_test['card1'] = train_test['card1'].fillna(-1)
train_test['card2'] = train_test['card2'].fillna(-1)
train_test['addr1'] = train_test['addr1'].fillna(-1)
train_test['P_emaildomain'] = train_test['P_emaildomain'].fillna(-1)
# train_test['TransactionDT'] = train_test['TransactionDT'].map(lambda x:(x//(3600*24*7)))



cache = train_test[['TransactionDT','TransactionAmt','uid','card2','addr1','P_emaildomain']].values
isFraud = train_test[['isFraud']].values



card1_time = {}
card2_time = {}
amt_time = {}
addr1_time = {}
# for i in range(cache.shape[0]):
#     time = int(cache[i,0]//3600)
#     amt = cache[i,1]
#     card1 = int(cache[i,2])
#     card2 = int(cache[i,3])
#     addr1 = int(cache[i,4])
    
#     if card1 != -1:
#         card1_time[card1] = card1_time.get(card1,{})
#         card1_time[card1][time] = card1_time[card1].get(time,[])
#         card1_time[card1][time].append(i)
#     if card2 != -1:
#         card2_time[card2] = card2_time.get(card2,{})
#         card2_time[card2][time] = card2_time[card2].get(time,[])
#         card2_time[card2][time].append(i)
#     amt_time[amt] = amt_time.get(amt,{})
#     amt_time[amt][time] = amt_time[amt].get(time,[])
#     amt_time[amt][time].append(i)
#     if addr1 != -1:
#         addr1_time[addr1] = addr1_time.get(addr1,{})
#         addr1_time[addr1][time] = addr1_time[addr1].get(time,[])
#         addr1_time[addr1][time].append(i)
    
#     if i % 100000 == 0:
#         print("*")
    
# print("-")


# f1 = open('../input/f3_deepwalk.csv','w')
# maxnums = 200000
# maxlength = 128
# for e in range(maxnums):
#     i = random.randint(0,cache.shape[0] - 1)

#     walked_node = []
#     resstr = []
#     length = 0
#     while(length < maxlength):
#         walked_node.append(i)
#         time = int(cache[i,0]//3600)
#         amt = cache[i,1]
#         card1 = int(cache[i,2])
#         card2 = int(cache[i,3])
#         addr1 = int(cache[i,4])
#         valid_i = []
#         for t in range(time - 3,time + 4):
#             for new_i in amt_time.get(amt,{}).get(t,[]):
#                 valid_i.append(("amt",new_i))
#             for new_i in card1_time.get(card1,{}).get(t,[]):
#                 valid_i.append(("card1_" + str(card1),new_i))
#             for new_i in card2_time.get(card2,{}).get(t,[]):
#                 valid_i.append(("card2_" + str(card2),new_i))
#             for new_i in addr1_time.get(addr1,{}).get(t,[]):
#                 valid_i.append(("addr1_" + str(addr1),new_i))
        
#         valid_i = [x for x in valid_i if x[1] not in walked_node]
#         if len(valid_i) == 0:
#             break
#         choice_i = random.randint(0,len(valid_i) - 1)
#         i = valid_i[choice_i][1]
#         if not valid_i[choice_i][0].startswith("amt") and (len(resstr) == 0 or resstr[-1] != valid_i[choice_i][0]):
#             resstr.append(valid_i[choice_i][0])
#         length += 1
        
    
#     if len(resstr) <= 1:
#         continue
#     f = f1
#     print(" ".join(resstr),file=f)
# #     print(" ".join(resstr))
# #     break
#     if e % 10000 == 0:
#         print("*")

# f1.close()

f1 = open('../input/f3_deepwalk.csv','w')
f = f1
index_feature = {}
index = 0
import pickle
# maxnums = 200000
# maxlength = 128
for i in range(cache.shape[0]):
    time = "TransactionDT_" + str(int(cache[i,0]//(3600*24*7)))
    time2 = "TransactionDT2_" + str(int((cache[i,0]+3600*24*3.5)//(3600*24*7)))
    amt = "amt_" + str(cache[i,1])
    card1 = "uid_" + str((cache[i,2]))
    card2 = "card2_" + str(int(cache[i,3]))
    addr1 = "addr1_" + str(int(cache[i,4]))
    P_emaildomain = "P_emaildomain_" + str(cache[i,5])
    
    if index_feature.get(time,-1) == -1:
        index_feature[time] = index
        index += 1  
    if index_feature.get(time2,-1) == -1:
        index_feature[time2] = index
        index += 1  
    if index_feature.get(card1,-1) == -1:
        index_feature[card1] = index
        index += 1
    if index_feature.get(addr1,-1) == -1:
        index_feature[addr1] = index
        index += 1
#     if index_feature.get(card2,-1) == -1:
#         index_feature[card2] = index
#         index += 1
    if index_feature.get(P_emaildomain,-1) == -1:
        index_feature[P_emaildomain] = index
        index += 1
    if index_feature.get(amt,-1) == -1:
        index_feature[amt] = index
        index += 1
        
    print("\t".join([str(index_feature[time]),str(index_feature[P_emaildomain])]),file=f)
    print("\t".join([str(index_feature[time]),str(index_feature[card1])]),file=f)
    print("\t".join([str(index_feature[time]),str(index_feature[amt])]),file=f)
    print("\t".join([str(index_feature[time]),str(index_feature[addr1])]),file=f)

    print("\t".join([str(index_feature[time2]),str(index_feature[P_emaildomain])]),file=f)
    print("\t".join([str(index_feature[time2]),str(index_feature[card1])]),file=f)
    print("\t".join([str(index_feature[time2]),str(index_feature[amt])]),file=f)
    print("\t".join([str(index_feature[time2]),str(index_feature[addr1])]),file=f)
    
    print("\t".join([str(index_feature[P_emaildomain]),str(index_feature[card1])]),file=f)
    print("\t".join([str(index_feature[P_emaildomain]),str(index_feature[amt])]),file=f)
    print("\t".join([str(index_feature[P_emaildomain]),str(index_feature[addr1])]),file=f)
    
    print("\t".join([str(index_feature[card1]),str(index_feature[amt])]),file=f)
    print("\t".join([str(index_feature[card1]),str(index_feature[addr1])]),file=f)

    print("\t".join([str(index_feature[amt]),str(index_feature[addr1])]),file=f)
    

f1.close()
print(len(index_feature))
with open('emb/index_feature.pkl', 'wb') as f:
    pickle.dump(index_feature, f)   


import os
os.system("python proNE.py -graph ../input/f3_deepwalk.csv -emb1 emb/sparse.emb -emb2 emb/spectral.emb -dimension 8 -step 5 -theta 0.5 -mu 0.2")
    
# import logging
# import gensim
# from gensim.models import Word2Vec


# dim = 8
# min_count = 2

# dump_name = './f3_w2v.{}.{}.gensim.txt'.format( dim, min_count)
# seg_corpus = []

# for line in open("../input/f3_deepwalk.csv"):
#     if not line:
#         continue
#     seg_corpus.append(line.strip().split())

# print(len(seg_corpus))
# print(" ".join(seg_corpus[0]))
# word2vec = Word2Vec(seg_corpus, size=dim, min_count=min_count, sg=1, hs=0, negative=10, iter=5, workers=6, window=5)
# word2vec.wv.save_word2vec_format(dump_name)