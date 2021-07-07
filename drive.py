import numpy as np
from util import function
from util import model
from util import function_plot

from scipy.integrate import simps
from scipy.stats import pearsonr
index = np.load("data/index_1.npy")
data_band = np.load("data/data_band_2.npy")
lai = np.load("data/lai.npy")






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

