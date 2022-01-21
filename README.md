MTGBM 一种多任务的GBDT实现
=========================================

简介  
-----------------------------

在树模型工业应用场景中，我们常常遇到需要多任务或多目标拟合的情况，在各种表格型特征比赛中多任务的场景也非常丰富，例如在推荐系统中我们预测广告点击率和转化率，或交通系统预测某一地点某一时间的人流和车流等。  


这些任务的特点是拥有多个有关联的标签可供学习，在神经网络中可以利用多头轻松解决这种情况，而树模型则只能利用不相关的树/森林去分别拟合这些标签。我们设计一种同构异值树，通过不同任务寻求每一个节点的梯度下降及分裂模式，获得更好泛化性的树，最后为树节点赋值完成一个森林。  



安装 & 样例
-----------------------------
我们的实现基于[LIGHTGBM](https://github.com/microsoft/LightGBM)，可以将我们的工程clone后使用  
```
cd python-package  
python setup.py install  
```
安装，注意需要卸载原版lightgbm进行使用，我们的api会兼容lightgbm的功能。

我们提供两个实例实验：  
[第一个实验](https://github.com/mtgbmcode/mtgbmcode/tree/main/examples/example1) 展示了对一段时间中外外汇量进行预测，我们将原本的单目标绝对交易量级转化为交易量级和交易量涨跌幅两个强相关且拥有不同意义的目标进行学习，在mape获得提升效果。  

[第二个实验](https://github.com/mtgbmcode/mtgbmcode/tree/main/examples/example2) 是kaggle的ieee比赛，在单目标二分类判断是否欺诈时，我们将不同的欺诈种类转化为多目标。注意这里的多目标并非多选一分类，而是多个二分类任务，在原生的LIGHTGBM与XGBOOST无法实现。

可以看到即使是单目标的数据集在经过合理构造后应用MT-GBM都可以获得一定提升。

进阶
-----------------------------

注意到我们的不同目标的梯度融合提供了默认的梯度相加实现，将相同意义的任务直接相加是合理的，例如点击率和转化率。但并非所有任务梯度都是处于同一个量级或相同的单调性意义，所以算法提供了自定义的梯度定义：
```python
def mypbj(preds, train_data, ep = 0):
    labels = train_data.get_label()
    labels2 = labels.reshape((num_labels,-1)).transpose()    
    preds2 = preds.reshape((num_labels,-1)).transpose()
    grad2 = (preds2 - labels2)                               
    grad = grad2 * np.array([1.5,0.001])     # 不同量级的梯度归一后相加                
    grad = np.sum(grad,axis = 1)
    grad2 = grad2.transpose().reshape((-1))                  
    hess = grad * 0. + 1
    hess2 = grad2 * 0. + 1
    return grad, hess, grad2, hess2    

clf=lgb.train(param,train_data,verbose_eval,
                fobj = mypbj,
                num_boost_round=100)
```  
理论上说通过自定义不同任务的梯度融合可以实现任意任务的多目标学习。    
更多实现细节可以参考  https://arxiv.org/abs/2201.06239                   
            
联系方式
-----------------------------   
如有使用问题或者进阶优化欢迎联系 zhenzhe.yzz@antgroup.com
