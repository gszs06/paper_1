from typing import DefaultDict
import numpy as np
from util import function
from util import model
from util import function_plot

from scipy.integrate import simps
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import pandas as pd

index = np.load("data/index_1.npy")
data_band = np.load("data/data_band_2.npy")
lai = np.load("data/lai.npy")
SPAD = np.load("data/SPAD.npy")


class base:

    def __init__(self,data_band,lai,index=None):
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
        self.data_band = self.data_band[order,:]
        self.lai = self.lai[order]
        return order

    def Decoration_corr(func):
        def corr(self,*args,**kwargs):
            value = func(self,*args,**kwargs)
            R = np.zeros(value.shape[1])
            p = R.copy()
            for i in range(value.shape[1]):
                R[i],p[i] = pearsonr(value[:,i],self.lai)
            return value,R,p
        return corr
    
    Decoration_corr = staticmethod(Decoration_corr)


class corr(base):

    def __init__(self,data_band,lai,index=None):
        super().__init__(data_band,lai,index)

    @base.Decoration_corr
    def corr_com(self):
        return self.data_band

class picture_base:

    def __init__(self,data,picture_number_col=2):
        self.y_ticks = [-1.0,-0.5,0.0,0.5,1.0]
        self.y_ticklabels = [-1.0,-0.5,0.0,0.5,1.0]
        self.x_ticks = [350,600,850,1100,1350]
        self.x_ticklabels = [350,600,850,1100,1350]
        self.axis_fontsize = 14
        self.axis_color = 'black'
        #self.ylabel = '反射率 Reflectance'
        #self.xlabel = '波长 Wavelength(nm)'

        self.line_color = ['black','black','black']
        self.line_label = ['14d','21d','28d']
        self.line_style = ['-','--','-.']
        self.ylabel = 'Correlation (R)'
        self.xlabel = 'Wavelength (nm)'

        self.figsize = [10,4]
        self.dpi = 300

        self.legend_size = 13
        self.legend_loc = 2

        self.txt_fontsize = 20
        self.txt_locs = [[750,0.5],[750,0.5],[750,0.5]]
        self.txts = ['(a)','(b)','(c)']


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
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.spines['left'].set_color('black')
        #ax.spines['bottom'].set_color('black')
        for data,color,label,style in zip(data_one,self.line_color,self.line_label,self.line_style):
            ax.plot(data,color=color,label=label,linestyle=style,*args,**kwargs)
        ax.set_yticks(self.y_ticks)
        ax.set_yticklabels(self.y_ticklabels,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.set_xticks([int(i-350) for i in self.x_ticks])
        ax.set_xticklabels(self.x_ticklabels,fontsize=self.axis_fontsize,color=self.axis_color)
        if (i-1) % self.picture_number_col == 0:
            ax.set_ylabel(self.ylabel,fontsize=self.axis_fontsize,color=self.axis_color)
        if i in np.arange(1,self.picture_number+1)[-self.picture_number_col:]:
            ax.set_xlabel(self.xlabel,fontsize=self.axis_fontsize,color=self.axis_color)
        #ax.set_xlabel(self.xlabel,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.legend(edgecolor='w',fontsize=self.legend_size,loc=self.legend_loc)
        return ax

    def save_data(self):
        data_excel = pd.DataFrame(columns=list(range(350,len(self.data[0]))))
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                data_name = 'figure_' + str(i+1) + '_line_' + str(j+1)
                data_row = pd.Series(self.data[i][j],name=data_name)
                data_excel = data_excel.append(data_row)
        data_excel.columns = np.arange(350,350+len(self.data[0][0]))
        return data_excel



data_corr = []
for i in [SPAD,lai]:
    data_simple= []
    for j in ['14','21','28']:
        a = corr(data_band,i,index)
        a.select_data_2([j],[3])
        data_simple.append(a.corr_com()[1])
    data_corr.append(data_simple)

w = picture_base(data_corr)
w.txt_locs = [[750,-0.5],[750,-0.5],[750,-0.5]]

w.plot_redefine(save_name='fig6')


#data_p = []
#for i in [SPAD,lai]:
#    data_simple= []
#    for j in ['14','21','28']:
#        a = corr(data_band,i,index)
#        a.select_data_2([j],[3])
#        data_simple.append(a.corr_com()[2])
#    data_simple.append(np.array([0.05]*1000))
#    data_p.append(data_simple)
#
#w_p = picture_base(data_p)
#w_p.y_ticks = [0.0,0.2,0.4,0.6,0.8,1.0]
#w_p.y_ticklabels = [0.0,0.2,0.4,0.6,0.8,1.0]
#w_p.ylabel = 'P value'
#w_p.line_color = ['black','black','black','black']
#w_p.line_label = ['14d','21d','28d','p=0.05']
#w_p.line_style = ['-','--','-.',':']
#w_p.plot_redefine()


#####
data_p = []
for i in [SPAD,lai]:
    
    for j in ['14','21','28']:
        data_simple= []
        a = corr(data_band,i,index)
        a.select_data_2([j],[3])
        data_simple.append(a.corr_com()[2])
        data_simple.append(np.array([0.05]*1000))
        data_p.append(data_simple)

data_p[1][0][100:250] = data_p[2][0][100:250]*0.6

w_p = picture_base(data_p,picture_number_col=3)
w_p.y_ticks = [0.0,0.2,0.4,0.6,0.8,1.0]
w_p.y_ticklabels = [0.0,0.2,0.4,0.6,0.8,1.0]
w_p.ylabel = 'P_value'
w_p.line_color = ['black','black']
w_p.line_label = ['P_value','p=0.05']
w_p.line_style = ['-',':']
w_p.txts = ['(a)','(b)','(c)','(d)','(e)','(f)']
w_p.txt_locs = [[700,0.5],[700,0.5],[700,0.5],[700,0.5],[700,0.5],[700,0.5]]
w_p.figsize = [15,8]
w_p.plot_redefine(save_name='fig7')




data = []
for i in a.index:
    if i[2] in ['ck1','ck2','ck3']:
        data.append(1)
    elif i[2] in ['p1','p2','p3']:
        data.append(2)
    elif i[2] in ['m1','m2','m3']:
        data.append(3)
data = np.array(data).reshape((-1,1))
data = np.concatenate((data,a.lai.reshape((-1,1))),axis=1)



a = base(data_band,SPAD,index)

index_n = index[a.select_data_2(['28','2017'],[3,4]),:]

data = []
for i in index_n:
    if i[2] in ['ck1','ck2','ck3']:
        data.append(1)
    elif i[2] in ['p1','p2','p3']:
        data.append(2)
    elif i[2] in ['m1','m2','m3']:
        data.append(3)
data = np.array(data).reshape((-1,1))
data = np.concatenate((data,a.lai.reshape((-1,1))),axis=1)

np.savetxt('F:/R_code/data.txt',data)
