from logging import Handler
import numpy as np
from numpy.lib.npyio import save
from rpy2.robjects.packages import data
from util import function
from util import model
from util import function_plot

from scipy.integrate import simps
from scipy.stats import pearsonr
index = np.load("data/index_1.npy")
data_band = np.load("data/data_band_2.npy")
lai = np.load("data/lai.npy")
SPAD = np.load("data/SPAD.npy")
data_sum_Enorm = np.load("F:/光谱数据/data_sum_Enorm.npy")
data_sum_Enorm[:,-3] = 1 - data_sum_Enorm[:,-3]


mask = np.ones(lai.shape,dtype=bool)
mask[[5,40,315]] = False
index = index[mask]
lai = lai[mask]
class base:

    def __init__(self,data_band,lai,index=None):
        self.data_band = data_band
        self.index = index
        self.lai = lai
    
    def set_index(self,index):
        self.index = index
        return self

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
        self.lai = self.lai[order]
        self.index = self.index[order]
        return self
    
    def create_fact(self):
        fact = []
        for i in range(self.index.shape[0]):
            if self.index[i,2] in ['ck1','ck2','ck3']:
                fact.append('1')
            elif self.index[i,2] in ['p1','p2','p3']:
                fact.append('2')
            elif self.index[i,2] in ['m1','m2','m3']:
                fact.append('3')
        self.fact = fact
        return self

    def statis_test(self):
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        from rpy2.robjects import Formula
        import pandas as pd
        car = importr("car")
        agricolae = importr("agricolae")
        aov = robjects.r["aov"]
        row = robjects.r["row.names"]
        result = []
        for i in range(self.data_band.shape[1]):
            value = self.data_band[:,i]
            fact = np.array(self.fact,dtype=str)
            d = {'c':robjects.FactorVector((fact)),'a':robjects.FloatVector((value))}
            data = robjects.DataFrame(d)
            oneway = aov(Formula("a~c"),data=data)
            out = agricolae.LSD_test(oneway,'c',alpha = 0.05,p_adj="bonferroni")
            mar = out.rx('groups')[0]
            rownamemar = row(mar)
            marker = mar.rx('groups')[0]
            mean_data = out.rx('means')[0][0]
            sd = out.rx('means')[0][1]

            char = pd.DataFrame()
            char['name'] = rownamemar
            char['marker'] = marker[0]
            char['mean'] = mean_data
            char['sd'] = sd
            result.append(char)
        self.result = result
        
        return self


a = [base(data_sum_Enorm,lai,index).select_data_2([i,j],[4,3]).create_fact().statis_test() for i,j in zip(['2016','2016','2016','2017','2017','2017'],['14','21','28','14','21','28'])]





import rpy2.robjects as robjects
import os