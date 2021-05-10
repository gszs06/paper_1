#Author: Tao ZHANG
#Date: 2021/5/10
#Introduction: 
#--------------------------------------------------------------------------------------------------
import os 
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class model_base:
    def __init__(self,X,y,times=50):
        if X.ndim == 1:
            self.X = X.reshape((-1,1))
        self.y = y
        self.times = times
        self.lens = X.shape[0]

    def random_sort(self):
        random_index = [i for i in range(self.X.shape[0])]
        np.random.shuffle(random_index)
        train_data = self.X[random_index[0:int(self.lens*0.7)],:]
        train_label = self.y[random_index[0:int(self.lens*0.7)]]
        test_data = self.X[random_index[int(self.lens*0.7):],:]
        test_label = self.y[random_index[int(self.lens*0.7):]]
        return train_data,train_label,test_data,test_label

    def func1(self,x,a,b):
        return a * x + b
        
    def func2(self,x,a,b,c):
        return a * np.exp(b * x) + c

    def func3(self,x,a,b,c):
        return a * np.log(b * x) + c

    def func4(self,x,a,b,c):
        return a * x**b + c
    
    def evaluation_system(func):
        def evaluation(self,*args,**kwargs):
            evaluation_data = np.zeros((self.times,3))
            models = []
            for i in range(self.times):
                evaluation_data[i,:],model = func(self,*args,**kwargs)
                models.append(model)
            return evaluation_data,models
        return evaluation
    
    @evaluation_system
    def fit(self,funcstr):
        if funcstr == 'line':
            funcstr = self.func1
        elif funcstr == 'exp':
            funcstr = self.func2
        elif funcstr == 'log':
            funcstr = self.func3
        elif funcstr == 'power':
            funcstr = self.func4
        train_data,train_label,test_data,test_label = self.random_sort()
        popt,pcov = curve_fit(funcstr,train_data,train_label)
        prodict_label = []
        for test_x,test_y in zip(test_data,test_label):
            prodict_y = funcstr(test_x,*popt)
            prodict_label.append(prodict_y)
        prodict_label = np.array(prodict_label)
        MAE = np.sum(np.abs(test_label - prodict_label)) / test_label.shape[0]
        RMSE = np.sqrt(np.sum((test_label - prodict_label)**2) / test_label.shape[0])
        R2 = r2_score(test_label,prodict_label)
        return [R2,MAE,RMSE],popt

    @evaluation_system
    def line_fit(self):
        train_data,train_label,test_data,test_label = self.random_sort()
        model_line = LinearRegression()
        model_line.fit(train_data,train_label)
        R2 = model_line.score(test_data, test_label)
        prodict_label = model_line.predict(test_data)
        MAE = np.sum(np.abs(test_label - prodict_label)) / test_label.shape[0]
        RMSE = np.sqrt(np.sum((test_label - prodict_label)**2) / test_label.shape[0])
        return [R2,MAE,RMSE],model_line



            



#construction ...
"""
提纲是：，建立一个大class 中主要的函数有，随机更新数据，模型循环评估，子函数为线性回归模型，对数模型，指数模型等
"""