MTGBM 一种多任务的GBDT实现
=========================================

[中文版文档](https://github.com/antmachineintelligence/mtgbmcode/blob/main/README.CH)

Despite the success of deep learning in computer vision and natural language processing, Gradient Boosted Decision Tree (GBDT) is yet one of the most powerful tools for applications with tabular data such as e-commerce and FinTech. However, applying GBDT to multi-task learning is still a challenge. Unlike deep models that can jointly learn a shared latent representation across multiple tasks, GBDT can hardly learn a shared tree structure. In this paper, we propose Multi-task Gradient Boosting Machine (MT-GBM), a GBDT-based method for multi-task learning. The MT-GBM can find the shared tree structures and split branches according to multi-task losses.


Installation and Examples
-----------------------------
Our project is implemented based on [LIGHTGBM](https://github.com/microsoft/LightGBM). Get start by:  
```
cd python-package  
python setup.py install  
```
Make sure that the old lightgbm has been uninstalled.  


We offer two examples which you can find the details in the paper：  
[Example I](https://github.com/mtgbmcode/mtgbmcode/tree/main/examples/example1) 
[Example II](https://github.com/mtgbmcode/mtgbmcode/tree/main/examples/example2) 



For different scenes, we recommend you to customize the loss function to get the best performance.
```python
def mypbj(preds, train_data, ep = 0):
    labels = train_data.get_label()
    labels2 = labels.reshape((num_labels,-1)).transpose()    
    preds2 = preds.reshape((num_labels,-1)).transpose()
    grad2 = (preds2 - labels2)                               
    grad = grad2 * np.array([1.5,0.001])                    
    grad = np.sum(grad,axis = 1)
    grad2 = grad2.transpose().reshape((-1))                  
    hess = grad * 0. + 1
    hess2 = grad2 * 0. + 1
    return grad, hess, grad2, hess2    

clf=lgb.train(param,train_data,verbose_eval,
                fobj = mypbj,
                num_boost_round=100)
```  
  

For more information see here https://arxiv.org/abs/2201.06239.                 
            
Support
-----------------------------   
Plz email [me](zhenzhe.yzz@antgroup.com) if you have any questions.
