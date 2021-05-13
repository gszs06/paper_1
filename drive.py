import numpy as np
from util import function

from scipy.integrate import simps
from scipy.stats import pearsonr
index = np.load("data/index_1.npy")
data_band = np.load("data/data_band_2.npy")
lai = np.load("data/lai.npy")



import matplotlib.pyplot as plt
import pandas as pd

data = [[data_band[1,:],data_band[2,:],data_band[3,:]],
        [data_band[4,:],data_band[5,:],data_band[6,:]],
        [data_band[7,:],data_band[8,:],data_band[9,:]]]

class picture_base:
    def __init__(self):
        self.y_ticklabels = [0,0.1,0.2,0.3,0.4,0.5,0.6]
        self.x_ticklabels = [350,600,850,1100,1350]
        self.axis_fontsize = 14
        self.axis_color = 'black'

        self.line_color = ['blue','green','red']
        self.line_label = ['T1','T2','T3']
        self.ylabel = '反射率 Reflectance'
        self.xlabel = '波长 Wavelength(nm)'

        self.figsize = [15,4]
        self.dpi = 200

        self.legend_size = 13
        self.legend_loc = 2

        self.txt_fontsize = 20
        self.txt_locs = [[0,0.3],[0,0.3],[0,0.3]]
        self.txts = ['(a)','(b)','(c)']

class picture(picture_base):

    def __init__(self,data,picture_number_col=3):
        super().__init__()
        self.data = data
        self.picture_number_col = picture_number_col
        self.picture_number = len(data)
        self.picture_number_row = int(np.ceil(self.picture_number / self.picture_number_col))


    def plot_redefine(self,save_name=None,*args,**kwargs):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=self.figsize,dpi=self.dpi)
        for i,txt_loc,txt in zip(range(1,self.picture_number+1),self.txt_locs,self.txts):
            ax = fig.add_subplot(self.picture_number_row,self.picture_number_col,i)
            ax = self.setting(ax,self.data[i-1],i)
            ax.text(*txt_loc,txt,fontsize=self.txt_fontsize)
        plt.tight_layout()
        if save_name:
            plt.savefig('output/'+save_name+'.png',dpi=200)
            figure_data = self.save_data()
            figure_data.to_excel('output/'+save_name+'_figdata.xlsx')
        plt.show()

    def setting(self,ax,data_one,i,*args,**kwargs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        for data,color,label in zip(data_one,self.line_color,self.line_label):
            ax.plot(data,color=color,label=label,*args,**kwargs)
        ax.set_yticks(self.y_ticklabels)
        ax.set_yticklabels(self.y_ticklabels,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.set_xticks([int(i-350) for i in self.x_ticklabels])
        ax.set_xticklabels(self.x_ticklabels,fontsize=self.axis_fontsize,color=self.axis_color)
        if (i-1) % self.picture_number_col == 0:
            ax.set_ylabel(self.ylabel,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.set_xlabel(self.xlabel,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.legend(edgecolor='w',fontsize=self.legend_size,loc=self.legend_loc)
        return ax

    def save_data(self):
        data_excel = pd.DataFrame(columns=list(range(350,len(data[0]))))
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                data_name = 'figure_' + str(i+1) + '_line_' + str(j+1)
                data_row = pd.Series(self.data[i][j],name=data_name)
                data_excel = data_excel.append(data_row)
        data_excel.columns = np.arange(350,350+len(self.data[0][0]))
        return data_excel




















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