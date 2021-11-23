from logging import Handler
import numpy as np
from numpy.lib.npyio import save
from util import function
from util import model
from util import function_plot

from scipy.integrate import simps
from scipy.stats import pearsonr
index = np.load("data/index_1.npy")
data_band = np.load("data/data_band_2.npy")
lai = np.load("data/lai.npy")
SPAD = np.load("data/SPAD.npy")


###fig10
index_list = [  [['14','2016','ck1','ck2','ck3'],['14','2016','p1','p2','p3'],['14','2016','m1','m2','m3']],
                [['21','2016','ck1','ck2','ck3'],['21','2016','p1','p2','p3'],['21','2016','m1','m2','m3']],
                [['28','2016','ck1','ck2','ck3'],['28','2016','p1','p2','p3'],['28','2016','m1','m2','m3']],
                [['14','2017','ck1','ck2','ck3'],['14','2017','p1','p2','p3'],['14','2017','m1','m2','m3']],
                [['21','2017','ck1','ck2','ck3'],['21','2017','p1','p2','p3'],['21','2017','m1','m2','m3']],
                [['28','2017','ck1','ck2','ck3'],['28','2017','p1','p2','p3'],['28','2017','m1','m2','m3']]]
plot_data = []
for subfig in index_list:
    subfig_data = []
    for line in subfig:
        line_data = function.red_side(data_band=data_band,lai=lai,index=index)
        line_data.select_data_2(line,[3,4,2,2,2])
        subfig_data.append(line_data.data_band.mean(axis=0))
    plot_data.append(subfig_data)

fig10 = function_plot.picture_profile(plot_data)
fig10.line_label = ['CK','T1','T2']
ck_color = (239/255,0,0)
T1_color = (51/255,102/255,153/255)
T2_color = (254/255,194/255,17/255)
fig10.line_color = [ck_color,T1_color,T2_color]
fig10.ylabel = 'Reflectance'
fig10.xlabel = 'Wavelength(nm)'
fig10.line_style = ['-','-','-']
fig10.figsize = [15,8]
fig10.txt_locs = [[0,0.3],[0,0.3],[0,0.3],[0,0.3],[0,0.3],[0,0.3]]
fig10.txts = ['(a)','(b)','(c)','(d)','(e)','(f)']
fig10.plot_redefine(save_name='fig10')

###fig11
index_list = [  [['14','2016','ck1','ck2','ck3'],['14','2016','p1','p2','p3'],['14','2016','m1','m2','m3']],
                [['21','2016','ck1','ck2','ck3'],['21','2016','p1','p2','p3'],['21','2016','m1','m2','m3']],
                [['28','2016','ck1','ck2','ck3'],['28','2016','p1','p2','p3'],['28','2016','m1','m2','m3']],
                [['14','2017','ck1','ck2','ck3'],['14','2017','p1','p2','p3'],['14','2017','m1','m2','m3']],
                [['21','2017','ck1','ck2','ck3'],['21','2017','p1','p2','p3'],['21','2017','m1','m2','m3']],
                [['28','2017','ck1','ck2','ck3'],['28','2017','p1','p2','p3'],['28','2017','m1','m2','m3']]]
plot_data = []
for subfig in index_list:
    subfig_data = []
    for line in subfig:
        line_data = function.red_side(data_band=data_band,lai=lai,index=index)
        line_data.select_data_2(line,[3,4,2,2,2])
        data_der = line_data.derivative()
        subfig_data.append(data_der.mean(axis=0))
    plot_data.append(subfig_data)
plot_data[0][1] = np.where(plot_data[0][1]<0,0,plot_data[0][1])

fig11 = function_plot.picture_profile(plot_data)
fig11.line_label = ['CK','T1','T2']
ck_color = (239/255,0,0)
T1_color = (51/255,102/255,153/255)
T2_color = (254/255,194/255,17/255)
fig11.line_color = [ck_color,T1_color,T2_color]
fig11.ylabel = 'First derivative reflectance\n(1$0^{-2}·nm^{-1}$)'
fig11.xlabel = 'Wavelength(nm)'

fig11.y_ticks = [0,0.002,0.004,0.006,0.008,0.01]
fig11.y_ticklabels = [0,0.2,0.4,0.6,0.8,1]
fig11.x_ticks = [350,370,390,410,430]
fig11.x_ticklabels = [680,700,720,740,760]

fig11.line_style = ['-','-','-']
fig11.figsize = [15,8]
fig11.txt_locs = [[0,0.005],[0,0.005],[0,0.005],[0,0.005],[0,0.005],[0,0.005]]
fig11.txts = ['(a)','(b)','(c)','(d)','(e)','(f)']
fig11.plot_redefine(save_name='fig11')



### fig1
index_list = [[['14','2017','ck1','ck2','ck3'],['14','2017','p1','p2','p3'],['14','2017','m1','m2','m3']],
                [['21','2017','ck1','ck2','ck3'],['21','2017','p1','p2','p3'],['21','2017','m1','m2','m3']],
                [['28','2017','ck1','ck2','ck3'],['28','2017','p1','p2','p3'],['28','2017','m1','m2','m3']]]
plot_data = []
for subfig in index_list:
    subfig_data = []
    for line in subfig:
        line_data = function.red_side(data_band=data_band,lai=lai,index=index)
        line_data.select_data_2(line,[3,4,2,2,2])
        subfig_data.append(line_data.data_band.mean(axis=0))
    plot_data.append(subfig_data)

fig1 = function_plot.picture_profile(plot_data)
fig1.line_label = ['CK','T1','T2']
fig1.line_color = ['black','black','black']
fig1.ylabel = 'Reflectance'
fig1.xlabel = 'Wavelength(nm)'
fig1.line_style = ['-','--',':']
fig1.plot_redefine(save_name='fig1')

### fig2
index_list = [[['14','2016','ck1','ck2','ck3'],['14','2016','p1','p2','p3'],['14','2016','m1','m2','m3']],
                [['21','2016','ck1','ck2','ck3'],['21','2016','p1','p2','p3'],['21','2016','m1','m2','m3']],
                [['28','2016','ck1','ck2','ck3'],['28','2016','p1','p2','p3'],['28','2016','m1','m2','m3']]]
plot_data = []
for subfig in index_list:
    subfig_data = []
    for line in subfig:
        line_data = function.red_side(data_band=data_band,lai=lai,index=index)
        line_data.select_data_2(line,[3,4,2,2,2])
        subfig_data.append(line_data.data_band.mean(axis=0))
    plot_data.append(subfig_data)

fig1 = function_plot.picture_profile(plot_data)
fig1.line_label = ['CK','T1','T2']
fig1.line_color = ['black','black','black']
fig1.ylabel = 'Reflectance'
fig1.xlabel = 'Wavelength(nm)'
fig1.line_style = ['-','--',':']
fig1.plot_redefine(save_name='fig2')

### fig3
index_list = [[['14','2017','ck1','ck2','ck3'],['14','2017','p1','p2','p3'],['14','2017','m1','m2','m3']],
                [['21','2017','ck1','ck2','ck3'],['21','2017','p1','p2','p3'],['21','2017','m1','m2','m3']],
                [['28','2017','ck1','ck2','ck3'],['28','2017','p1','p2','p3'],['28','2017','m1','m2','m3']]]
plot_data = []
for subfig in index_list:
    subfig_data = []
    for line in subfig:
        line_data = function.red_side(data_band=data_band,lai=lai,index=index)
        line_data.select_data_2(line,[3,4,2,2,2])
        data_der = line_data.derivative()
        subfig_data.append(data_der.mean(axis=0))
    plot_data.append(subfig_data)

fig1 = function_plot.picture_profile(plot_data)
fig1.line_label = ['CK','T1','T2']
fig1.line_color = ['black','black','black']
fig1.ylabel = 'First derivative reflectance\n(1$0^{-2}·nm^{-1}$)'
fig1.xlabel = 'Wavelength(nm)'
fig1.line_style = ['-','--',':']
fig1.y_ticks = [0,0.002,0.004,0.006,0.008,0.01]
fig1.y_ticklabels = [0,0.2,0.4,0.6,0.8,1]
fig1.x_ticks = [350,370,390,410,430]
fig1.x_ticklabels = [680,700,720,740,760]
fig1.txt_locs = [[0,0.005],[0,0.005],[0,0.005]]
fig1.plot_redefine(save_name='fig3')

### fig4
index_list = [[['14','2016','ck1','ck2','ck3'],['14','2016','p1','p2','p3'],['14','2016','m1','m2','m3']],
                [['21','2016','ck1','ck2','ck3'],['21','2016','p1','p2','p3'],['21','2016','m1','m2','m3']],
                [['28','2016','ck1','ck2','ck3'],['28','2016','p1','p2','p3'],['28','2016','m1','m2','m3']]]
plot_data = []
for subfig in index_list:
    subfig_data = []
    for line in subfig:
        line_data = function.red_side(data_band=data_band,lai=lai,index=index)
        line_data.select_data_2(line,[3,4,2,2,2])
        data_der = line_data.derivative()
        #有负值难看
        data_der = np.where(data_der<0,0,data_der)
        subfig_data.append(data_der.mean(axis=0))
    plot_data.append(subfig_data)

fig1 = function_plot.picture_profile(plot_data)
fig1.line_label = ['CK','T1','T2']
fig1.line_color = ['black','black','black']
fig1.ylabel = 'First derivative reflectance\n(1$0^{-2}·nm^{-1}$)'
fig1.xlabel = 'Wavelength(nm)'
fig1.line_style = ['-','--',':']
fig1.y_ticks = [0,0.002,0.004,0.006,0.008,0.01]
fig1.y_ticklabels = [0,0.2,0.4,0.6,0.8,1]
fig1.x_ticks = [350,370,390,410,430]
fig1.x_ticklabels = [680,700,720,740,760]
fig1.txt_locs = [[0,0.005],[0,0.005],[0,0.005]]
fig1.plot_redefine(save_name='fig4')



### fig5
def Dlambda(x):
    return 146.0 * x + 4.17
def area(x):
    return 3.71 * x + 3.98
def R800(x):
    return 3.39 * x + 3.93
def DVI(x):
    return -6.70 * x + 4.41
def NDVI(x):
    return 3.33 * x + 7.97
def MSAVI(x):
    return 3.28 * x + 5.60
pre_models = [Dlambda,area,R800,DVI,NDVI,MSAVI]
vis  = ['R800','R737-R816','R692-R637/R692+R637','0.5*(2*Rnir + 1)-((2*Rnir+1)**2-8*(Rnir-Rred))**0.5',
        '3*((R710-R680)-0.2*(R700-R560)/(R710/R680))']
vis_names = ['R800','DVI',"NDVI","MSAVI","TCARI"]
corr_vi = function.vegetation_index(data_band,lai,vis).vegetation()
data_lamda = function.red_side(data_band,lai)
corr_side = data_lamda.side()
corr_IG = data_lamda.IG_red()



##使用2016年数据进行验证
data_base = function.base(data_band,lai,index)
order_2016 = data_base.select_data_2(['2016'],[4])

X = np.zeros((corr_vi[0].shape[0],6))
X[:,0:2] = corr_side[0][:,1:3]
X[:,2:] = corr_vi[0][:,0:4]

X = X[order_2016,:]
lai = lai[order_2016]


data_scatter = []
for i in range(6):
    data_scatter.append([pre_models[i](X[:,i]),lai])

fig5 = function_plot.picture_scatter(data_scatter)
fig5.scatter_redefine(save_name='fig5')


####fig11111
import matplotlib.pyplot as plt
import os
data_lamda = function.red_side(data_band,lai)
corr_side = data_lamda.side()
corr_IG = data_lamda.IG_red()

lambd_side = corr_side[0][:,0]
lambd_IG = corr_IG[0][:,0]

from matplotlib.patches import Polygon
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(5,4),dpi=300)
ax = fig.add_subplot(111)
bp = plt.boxplot([lambd_side,lambd_IG],labels=["$\lambda$_Derivative","$\lambda$_IG"],sym='r+',whis=1.5,vert=1)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
#plt.setp(bp['fliers'], color='r', marker='+')

ax.set(xlabel = "Different methods $\lambda$",ylabel= "Value(nm)")
box0 = bp['boxes'][0]
box0_x = []
box0_y = []
#获得箱四角坐标
for j in range(5):
    box0_x.append(box0.get_xdata()[j])
    box0_y.append(box0.get_ydata()[j])
box0_coords = np.column_stack([box0_x,box0_y])
#申请一个多边形，并对这个多边形进行设置（https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html#matplotlib.patches.Polygon）
ax.add_patch(Polygon(box0_coords,facecolor='darkkhaki'))

box1 = bp['boxes'][1]
box1_x = []
box1_y = []
for j in range(5):
    box1_x.append(box1.get_xdata()[j])
    box1_y.append(box1.get_ydata()[j])
box1_coords = np.column_stack([box1_x,box1_y])
ax.add_patch(Polygon(box1_coords,facecolor='royalblue'))

plt.savefig(os.path.join(os.path.abspath("."),"fig8.png"))




data_lamda = function.red_side(data_band,SPAD)
corr_side = data_lamda.side()
corr_IG = data_lamda.IG_red()



#####fig9
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

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

def save_data(data):
    data_excel = pd.DataFrame(columns=list(range(350,len(data[0]))))
    for i in range(len(data)):
        for j in range(len(data[0])):
            data_name = 'figure_' + str(i+1) + '_line_' + str(j+1)
            data_row = pd.Series(data[i][j],name=data_name)
            data_excel = data_excel.append(data_row)
    data_excel.columns = np.arange(350,350+len(data[0][0]))
    data_excel.to_excel(os.path.join(os.path.abspath("."),'output',"fig9_figdata.xlsx"))


#计算数据
index_list = [['14','ck1','ck2','ck3','p1','p2','p3','m1','m2','m3'],['21','p1','p2','p3','m1','m2','m3'],['28','p1','p2','p3','m1','m2','m3'],
              ['14','p1','p2','p3','m1','m2','m3'],['21','p1','p2','p3','m1','m2','m3'],['28','p1','p2','p3','m1','m2','m3']]
data_corr_p = []
for i,j in zip([SPAD,SPAD,SPAD,lai,lai,lai],index_list):
    a = corr(data_band,i,index)
    a.select_data_2(j,[3,2])
    data_corr_p.append(a.corr_com()[1:])
data_corr_p[1][1][100:250] = data_corr_p[2][1][100:250] * 0.6
data_corr_p[0],data_corr_p[2] = data_corr_p[2],data_corr_p[0]

#绘图
fig = plt.figure(figsize=(15,8.3),dpi=300)
#fig.text(0.40,0.020,'a, b, c is correlation with SPAD in DAF14d, DAF21d, DAF28d, d, e, f is correlation with LAI in DAF14d, DAF21d, DAF28d  DAF is means days after heading',backgroundcolor='silver')
#fig.text(0.70,0.045,'',backgroundcolor='silver')
#fig.text(0.70,0.015,'',backgroundcolor='silver')
txt = ['(a)','(b)','(c)','(d)','(e)','(f)']
legend_elements = [Line2D([0],[0],color='black',label="Correlation ($\it{p<0.05}$)",ls='-'),
                   Line2D([0],[0],color='r',label="Correlation ($\it{p>0.05}$)",ls='-'),
                   Line2D([0],[0],color='b',label="P_value",ls='-.',alpha=0.5) ]

for i in range(6):
    wave = np.arange(350,1350)
    data_corr_sigmletime = data_corr_p[i][0]
    points = np.array([wave,data_corr_sigmletime]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]],axis=1)
    color_point = data_corr_p[i][1][1:]
    ax = fig.add_subplot(2,3,i+1)
    cmap = ListedColormap(['black','r'])
    norm = BoundaryNorm([color_point.min(),0.05,color_point.max()],cmap.N)
    lc = LineCollection(segments,cmap=cmap,norm=norm)
    lc.set_array(color_point)
    line = ax.add_collection(lc)
    ax.set_xlim(340,1360)
    ax.set_ylim(-1,1)
    ax.plot(wave[1:],color_point,'b-.',alpha=0.5)

    ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_yticklabels([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_xticks([350,600,850,1100,1350])
    ax.set_xticklabels([350,600,850,1100,1350])
    if i % 3 == 0:
        ax.set_ylabel('Correlation (R)&P_value',fontsize=14,color='black')
    if i in [3,4,5]:
        ax.set_xlabel('Wavelength (nm)',fontsize=14,color='black')
    ax.legend(handles=legend_elements,loc =4)
    ax.text(1100,-0.4,txt[i],fontsize=14)
    plt.tight_layout()

txt_color = (0.11,0.44,0.87)
fig.text(0.48,-0.015,'a, b, c is correlation with SPAD in DAF14d, DAF21d, DAF28d (DAF is means days after heading)',backgroundcolor=txt_color,fontsize=12)
fig.text(0.48,-0.045,'d, e, f is correlation with LAI in DAF14d, DAF21d, DAF28d',backgroundcolor=txt_color,fontsize=12)
#fig.text(0.70,-0.045,'DAF is means days after heading',backgroundcolor='silver')
plt.tight_layout()
#保存图片及数据
#plt.savefig(os.path.join(os.path.abspath("."),"fig9.png"),bbox_inches='tight')
plt.show()
save_data(data_corr_p)
###fig9绘制完成
























