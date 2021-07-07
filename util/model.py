#Author: Tao ZHANG
#Date: 2021/5/10
#Introduction: 对已经计算出的高相关光谱指数，建立光谱指数与植被指数的相关模型（70%训练，30%验证），并得出评估结果
#Coding ideas: 建立一个大class 中主要的函数有，随机更新数据，模型循环评估（作为装饰器），子函数为线性回归模型，对
#              数模型，指数模型等
#------------------------------------------------------------------------------------------------------------
import os 
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class model_base:
    """
    说明：将高相关光谱指数数据和相应的植被指数数据导入，并将数据随机按照7：3比例分为训练集和测试集，通过传入特定的
          拟合函数进行拟合模型，并通过多次拟合得到最优的评估参数，评估参数包括RMSE，MAE和R2
    属性：
        X:     类型为array，维度为2（如果数据1维会切换为2维），为高相关光谱指数数据，如每条高光谱数据计算得出的红边位
                 置（假如红边位置与植物生理指数高相关）
        y:     类型为array，维度为1，为植物生理指数
        times: 类型为int，为循环拟合模型的次数，默认为50
    内置函数：
        >random_sort:
                    对所有数据X，y进行随机排列，并按照7：3将数据划分为训练集和测试集
                    input:
                        None
                    output:
                        train_data:  类型为array，维度为2，X的训练集
                        train_label: 类型为array，维度为1，y的训练集
                        test_data:   类型为array，维度为2，X的测试集
                        test_label： 类型为array，维度为1，y的测试集
        >func1:
                    简单的线性拟合方程，当fit函数中的funcstr参数为line时使用该函数进行拟合
                    input:
                        x:      一次线性方程的未知变量
                        a,b:    线性方程的参数a为斜率，b为截距
                    output:
                        返回一次线性方程的计算结果
        >func2:
                    一次指数函数，形式为a*np.exp(b*x)+c，当fit函数中funcstr参数为exp时使用此函数进行拟合
                    input:
                        x:      一次指数函数的未知变量
                        a,b,c:  指数函数的参数
                    output:
                        返回一次指数函数的计算结果 
        >func3:
                    一次对数函数，形式为a*np.log(b*x)+c，当fit函数中funcstr参数为log时使用此函数进行拟合
                    input:
                        x:      一次对数函数的未知变量
                        a,b,c:  对数函数的参数
                    output:
                        返回一次对数函数的计算结果
        >func4:
                    幂函数，形式为a*x**b+c，当fit函数中funcstr参数为power时使用此函数进行拟合
                    input:
                        x:      幂函数的未知变量
                        a,b,c:  幂函数的参数
                    output:
                        返回幂函数的计算结果
        >evaluation_system:
                    装饰器函数，对一个公式进行多次（times）拟合，并返回多次拟合后的评估参数和模型结果
        >fit:
                    给出一个具体拟合函数（函数定义要求如下），通过最小化得出公式的各个参数，得到具体拟合函数后
                        将测试数据代入计算得出评估参数RMSE，MAE和R2（计算公式如下）
                    input:
                        funcstr: 类型为字符串或者自定义一个形如f(x,...)的函数变量，字符串当为line,exp,log,power时分
                            别传入func1(线性，a * x + b),func2(指数，a * np.exp(b * x) + c),func3(对数，a * np.lo
                            g(b * x) + c),func4(幂，a * x**b + c)
                            自定义函数要求：需要将第一个形参变为x，要拟合的参数作为其余的形参，如要拟合逻辑方程形如
                                            a/(b + c*exp(-dx))，则定义的函数形式为
                                            def logistic(x,a,b,c,d):
                                                return a/(b + c*exp(-dx))
                            评估参数的计算公式：
                                            本代码中局限只选择了三个评估参数RMSE，MAE及R2，评估模型只能使用这三个参
                                                数，具体计算公式百度
                    output:
                        [R2,MAE,RMSE]: 类型为列表，使用测试数据计算所得的三个评估参数
                        popt:          类型为array，拟合最优模型的参数结果，按照传入形参的顺序得出
                        注1，由于该函数被evaluation_system函数装饰，因此会返回多次拟合的结果，其中评估参数为矩阵，
                            每一列分别为R2,MAE和RMSE，参数结果为list每一次拟合的参数结果
                        注2，此函数只能对含有一个变量的公式进行拟合，不能对多变量的公式进行拟合（其实也可以，假如
                            有三个变量，则在编写时分别为用x[0],x[1],x[2]来表示三个变量，但是要注意多变量间的相关性，
                            使用逐步回归或者数据降维）
        >line_fit:
                    基于机器学习中的最小二乘法回归训练线性回归模型（可以是多变量），返回评估结果和训练完成的模型
                    input:
                        None
                    output:
                        [R2,MAE,RMSE]: 类型为列表，使用测试数据计算所得的三个评估参数
                        model_line:    为sklear中模型对象
                    注1，使用了evaluation_system装饰，同上
    栗子：
        >> import numpy as np
        >> from scipy.optimize import curve_fit
        >> from sklearn.metrics import r2_score
        >> from sklearn.linear_model import LinearRegression
        # data X has been seleted out; data y existence
        >> a = model_base(X,y)
        >> evaluation_data,model_parameter = a.fit(funcstr='line') 
    """
    def __init__(self,X,y,times=50,index=None):
        if X.ndim == 1:
            self.X = X.reshape((-1,1))
        self.y = y
        self.times = times
        self.index = index
        self.lens = X.shape[0]
    def set_index(self,index):
        self.index = index

    def select_data_2(self,strlist,location):
        order = []
        for i in range(self.index.shape[0]):
            j = 0
            for loc in location:
                if self.index[i,int(loc)] in strlist:
                    j = j + 1
            if j == len(location):
                order.append(i)
        order = np.array(order,dtype=np.int)
        self.data_band = self.data_band[order,:]
        self.lai = self.lai[order,:]
        return order        

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