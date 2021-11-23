"""
提纲是：原始光谱曲线的绘图，一阶导数的绘图，模型的散点图，绘图数据保存为excel

"""
#Author: Tao ZHANG
#Date: 2021/5/13
#Introduction: 原始光谱曲线的绘图，一阶导数的绘图，模型的散点图，绘图数据保存为excel
#Coding ideas: 将图像的设置参数与图像的绘制过程分离，需要调整时只修改参数即可
#------------------------------------------------------------------------------------------------------------
import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt

class picture_base:
    """
    说明：绘图基本类，定义了基本的绘图参数作为父类，可以通过子类实例化后修改来设置图片属性
    属性：
                <坐标轴>
        y_ticklabels：类型为list，纵坐标轴的显示标签
        x_ticklabels：类型为list，横坐标轴的显示标签
        axis_fontsize：类型为int，坐标轴标签字体大小
        axis_color：类型为str，坐标轴标签字体颜色
        ylabel：类型为str，纵坐标轴名称
        xlabel：类型为str，横坐标轴名称
        
                <曲线>
        line_color：类型为list，长度与图中曲线数量相等，为图中各个曲线的颜色
        line_label：类型为list，长度与图中曲线数量相等，为图中各个曲线的标签

                <画布>
        figsize：类型为list，含两个元素，是画布的高宽，一般来将图高宽比为4：5，建议参数按此比例等比变化
        dpi：类型为int，为图片分辨率大小，一般为200

                <图例>
        legend_size：类型为int，为图例上字体大小
        legend_loc：类型为int，为图例出现在图片的位置，具体见plot函数的官方说明文档

                <注释文档>
        txt_fontsize：类型为int，为注释文档的字体大小
        txt_locs：类型为list，为每幅图注释文档的坐标位置，按照已经定义的坐标轴坐标定义
        txts：类型为list，为每幅图中的注释文档内容
    """
    def __init__(self):
        self.y_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6]
        self.y_ticklabels = [0,0.1,0.2,0.3,0.4,0.5,0.6]
        self.x_ticks = [350,600,850,1100,1350]
        self.x_ticklabels = [350,600,850,1100,1350]
        self.axis_fontsize = 14
        self.axis_color = 'black'
        self.ylabel = '反射率 Reflectance'
        self.xlabel = '波长 Wavelength(nm)'

        self.line_color = ['blue','green','red']
        self.line_label = ['T1','T2','T3']
        self.line_style = ['-','--','-.']
        self.ylabel = '反射率 Reflectance'
        self.xlabel = '波长 Wavelength(nm)'

        self.figsize = [15,4]
        self.dpi = 300

        self.legend_size = 13
        self.legend_loc = 2

        self.txt_fontsize = 20
        self.txt_locs = [[0,0.3],[0,0.3],[0,0.3]]
        self.txts = ['(a)','(b)','(c)']

class picture_profile(picture_base):
    """
    说明：读取相应的光谱数据进行绘图，并将绘图数据保存（默认的保存路径为/output/...）
    属性：
        父类picture_base中的所有绘图参数
        data：类型为list，具体的格式如下
                data数据采用了双list嵌套的形式，即形如[[第1张图数据],[第2张图数据],...,[第n张图数据],...]
                其中每张图数据作为一个list包含了若干曲线数据，即：图数据=[第1条曲线数据,第2条曲线数据...]
                因此绘制图片数量为len(data)，图中包含的曲线为len(data[0])
        picture_number_col：类型为int，为图片排列具体的列数，决定图片在画布上的排列方式，默认为3，如总图片
                            数为4时则会将画布划分为2*3大小
    内置函数：
    >plot_redefine:
                    绘制图片，如果传入名称则将所绘制图片及数据保存
                    input:
                        save_name：类型为str，默认为None，为图片保存名称，传入时将会将所绘图保存到本环境
                                output文件夹下，同时也会将绘图数据保存到该文件夹下的excel中，曲线数据逐
                                行保存，第一列为每条曲线数据的说明，如fig_1_line_1就是指第一张图中的第一
                                条曲线的数据
                    output:
                        None
    >setting:
                    plot_redefine函数设置子图图要素时调用，主要设置了以下几个方面：
                        1、上右坐标轴不出现
                        2、设置纵坐标及横坐标轴标签
                        3、设置坐标轴的名称
                        4、设置了图例
                    input:
                        ax：为一个已经建立完成的子图对象
                        data_one：为建立的子图对象中要绘制曲线的数据
                        i：子图对象在整个画布中的位置
                    output:
                        None
    >save_data:
                    plot_redefine函数保存绘图数据时调用，借助了pandas中的DataFrame数据格式
                    input:
                        None
                    output:
                        None
    栗子：
        >> import numpy as np
        >> import pandas as pd  
        >> import matplotlib.pyplot as plt
        # data has being
        >> a = picture(data)
        >> a.plot_redefine(save_name='figure1')
        # running code, the picture and data will appear in path "/output/figure1.png" and
        # "/output/figure1_figdata.xlsx"
    """
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


class picture_scatter:
    """
    说明：绘制散点图
    属性：
                <点属性>
        c: 类型为str,点的颜色
        s: 类型为int,点的大小
  
                <坐标轴>
        y_ticks：类型为list，纵坐标要定义标签的刻度
        y_ticklabels：类型为list，纵坐标轴的显示标签
        X_ticks：类型为list，横坐标要定义标签的刻度
        x_ticklabels：类型为list，横坐标轴的显示标签
        axis_fontsize：类型为int，坐标轴标签字体大小
        axis_color：类型为str，坐标轴标签字体颜色
        ylabel：类型为str，纵坐标轴名称
        xlabel：类型为str，横坐标轴名称        
    
                <画布>
        figsize：类型为list，含两个元素，是画布的高宽，一般来将图高宽比为4：5，建议参数按此比例等比变化
        dpi：类型为int，为图片分辨率大小，一般为300
 
                <图例>
        legend_size：类型为int，为图例上字体大小
        legend_loc：类型为int，为图例出现在图片的位置，具体见plot函数的官方说明文档

                <注释文档>
        txt_fontsize：类型为int，为注释文档的字体大小
        txt_locs：类型为list，为每幅图注释文档的坐标位置，按照已经定义的坐标轴坐标定义
        txts：类型为list，为每幅图中的注释文档内容
    内置函数：
    >scatter_redefine:
                    绘制图片，如果传入名称则将所绘制图片及数据保存
                    input:
                        save_name：类型为str，默认为None，为图片保存名称，传入时将会将所绘图保存到本环境
                                output文件夹下，同时也会将绘图数据保存到该文件夹下的excel中，曲线数据逐
                                行保存，第一列为每条曲线数据的说明，如fig_1_line_1就是指第一张图中的第一
                                条曲线的数据
                    output:
                        None
    >setting:
                    plot_redefine函数设置子图图要素时调用，主要设置了以下几个方面：
                        1、上右坐标轴不出现
                        2、设置纵坐标及横坐标轴标签
                        3、设置坐标轴的名称
                        4、设置了图例
                    input:
                        ax：为一个已经建立完成的子图对象
                        data_one：为建立的子图对象中要绘制曲线的数据
                        i：子图对象在整个画布中的位置
                    output:
                        None
    >save_data:
                    plot_redefine函数保存绘图数据时调用，借助了pandas中的DataFrame数据格式
                    input:
                        None
                    output:
                        None
    """
    def __init__(self,data,picture_number_col=3) -> None:
        self.data = data
        self.picture_number_col = picture_number_col
        self.picture_number = len(data)
        self.picture_number_row = int(np.ceil(self.picture_number / self.picture_number_col))

        self.c = 'black'
        self.s = 5

        self.y_ticks = [4.6,5.0,5.4,5.8]
        self.y_ticklabels = [4.6,5.0,5.4,5.8]
        self.x_ticks = [4.6,5.0,5.4,5.8]
        self.x_ticklabels = [4.6,5.0,5.4,5.8]
        self.axis_fontsize = 14  
        self.axis_color = 'black' 
        self.ylabel = 'LAI simulation ($m^{2}/m^{2}$)'
        self.xlabel = 'LAI measurement ($m^{2}/m^{2}$)'            

        self.figsize = [15,8]
        self.dpi = 300

        self.legend_size = 13
        self.legend_loc = 2

        self.txt_fontsize = 13
        self.txt_locs = [[4.6,5.45],[4.6,5.45],[4.6,5.45],[4.6,5.45],[4.6,5.45],[4.6,5.45]]
        self.txts = ['(a) D$\lambda$\ny = 146.0x + 4.17\n$R^2$ = 0.59','(b) area\ny = 3.71x + 3.98\n$R^2$ = 0.65*',
                    '(c) R(800)\ny = 3.39x + 3.93\n$R^2$ = 0.64*','(d) DVI(737,816)\ny = -6.70x + 4.41\n$R^2$ = 0.80**',
                    '(e) NDVI(692,637)\ny = 3.33x + 7.97\n$R^2$ = 0.75**','(f) MSAVI\ny = 3.28x + 5.60\n$R^2$ = 0.66*']

    def scatter_redefine(self,save_name=None,*args,**kwargs):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=self.figsize,dpi=self.dpi)
        for i,txt_loc,txt in zip(range(1,self.picture_number+1),self.txt_locs,self.txts):
            ax = fig.add_subplot(self.picture_number_row,self.picture_number_col,i)
            ax = self.setting(ax,self.data[i-1],i)
            ax.text(*txt_loc,txt,fontsize=self.txt_fontsize)
        plt.tight_layout()
        if save_name:
            plt.savefig('output/'+save_name+'.png',dpi=self.dpi)
            figure_data = self.save_data()
            figure_data.to_excel('output/'+save_name+'_figdata.xlsx')
        plt.show()

    def setting(self,ax,data_one,i,*args,**kwargs):
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.scatter(*data_one,color=self.c,s=self.s)
        line = np.linspace(np.min(data_one),np.max(data_one),num=50,endpoint=True)
        ax.plot(line,line,color='black',label='1:1 line')
        ax.set_yticks(self.y_ticks)
        ax.set_yticklabels(self.y_ticklabels,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.set_xticks(self.x_ticks)
        ax.set_xticklabels(self.x_ticklabels,fontsize=self.axis_fontsize,color=self.axis_color)
        if (i-1) % self.picture_number_col == 0:
            ax.set_ylabel(self.ylabel,fontsize=self.axis_fontsize,color=self.axis_color)
        #先写死之后再想办法
        if i in np.arange(1,self.picture_number+1)[-self.picture_number_col:]:
            ax.set_xlabel(self.xlabel,fontsize=self.axis_fontsize,color=self.axis_color)
        ax.legend(edgecolor='w',fontsize=self.legend_size,loc=self.legend_loc)
        return ax    

    def save_data(self):
        data_excel = pd.DataFrame(columns=list(range(len(self.data[0][0]))))
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                data_name = 'figure_' + str(i+1) + '_line_' + str(j+1)
                data_row = pd.Series(self.data[i][j],name=data_name)
                data_excel = data_excel.append(data_row)
        data_excel.columns = np.arange(len(self.data[0][0]))
        return data_excel                              
        