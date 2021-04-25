import numpy as np
from util import function

from scipy.integrate import simps
from scipy.stats import pearsonr
index = np.load("data/index_1.npy")
data_band = np.load("data/data_band_2.npy")
lai = np.load("data/lai.npy")





def dec(func):
    def a(*args,**kwargs):
        value = func(*args,**kwargs)
        return value + 1
    return a

@dec
def add(m,n):
    w = m+n
    return w

import numpy as np

class a:
    b = 1
    def __init__(self,a1):
        self.a1 = a1
        print(a.b)


class b(a):
    def __init__(self,a1,b1):
        super().__init__(a1)
        self.b1 = b1
    def add(self):
        print(self.a1 + self.b1)
index = np.load("F:/光谱数据/index_1.npy")

class Decor(object):
    def __init__(self,func):
        self.func = func
    def __call__(self,*args):
            w = self.func(*args)
            return w+1

class De(object):
    def __init__(self,a):
        self.a = a
    @Decor
    def add(self):
        return self.a

@Decor
def sum(a,b):
    return a+b


class c:
    def __init__(self,w):
        self.w = w
    def dec1(func):
        def fun(self,*args,**kwargs):
            value = func(self,*args,**kwargs)
            return value+self.w
        return fun
    def suma(self,a,b):
        return a+b
    @dec1
    def suma1(self,a,b):
        return a+b
class d(c):
    def __init__(self,w):
        super().__init__(w)
    


def dec(func):
    def a(self,*args,**kwargs):
        value = func(self,*args,**kwargs)
        return value + self.w
    return a


class base:
 
    def __init__(self,data_band,lai,lamda1=680,lamda2=760,index=None):
        self.lamda1 = int(lamda1)
        self.lamda2 = int(lamda2)
        self.data_band = data_band
        self.index = index
        self.lai = lai
    
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
        return order
    def Decoration_corr(func):
        def corr(self,*args,**kwargs):
            value = func(self,*args,**kwargs)
            R = np.zeros(value.shape[1])
            p = R.copy()
            for i in range(value.shape[1]):
                R[i],p[i] = pearsonr(value[:,i],self.lai)
            return R,p
        return corr
    @Decoration_corr
    def IG_red(self):
        Rs = np.mean(self.data_band[:,780-350:795-350],axis=1)
        Rs = Rs.reshape((-1,1))
        Ro = np.mean(self.data_band[:,670-350:675-350],axis=1)
        Ro = Ro.reshape((-1,1))
        Wave = np.arange(350,1350)
        y = self.data_band[:,685-350:780-350]
        x = Wave[685-350:780-350]
        y = np.sqrt(-np.log((Rs - y) / (Rs - Ro)))
        w = np.polyfit(x,y.T,1)
        lamda0 = -w[1,:] / w[0,:]
        sigma = 1.0 / np.sqrt(2*w[0,:])
        IG_data = np.concatenate((lamda0.reshape((-1,1)),sigma.reshape((-1,1))),axis=1)
        return IG_data    

def func(**b):
    for i in b:
        print(i)