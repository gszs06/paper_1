#Author: Tao ZHANG
#Date: 2021/4/20
#Introduction: 定义了高光谱数据相关计算的一些类，主要包括了基础数据类，红边参数计算类，植被指数计算类 
#--------------------------------------------------------------------------------------------------
import os 
import numpy as np
from scipy.integrate import simps
from scipy.stats import pearsonr
from sympy import sympify,Symbol

class base:
    jls_extract_var = """
        说明：定义数据的基础类用于保存光谱、生理指数及索引数据，及设定处理编号，根据处理筛选数据等函数
        属性：
            data_band:  类型为array，维度为2，每一条光谱数据逐行保存
            lai:        类型为array，维度为1，每一条光谱数据对应的生理指数数据
            index:      可忽略，类型为array，维度为2，为每一条光谱数据对应的试验编号如（“年”，“生育期”，“处理名称”）
                            不存在时不能使用set_index和select_data_2函数
        内置函数：
            >set_index:
                        对逐条光谱数据设置试验编号,如“编号1、编号2、处理名称、生育期、年份”等 
                        input: 
                            index: 类型为array，维度为2，试验编号数据
                        output:
                            None
            >select_data_2:
                        按照试验编号数据对光谱数据进行选择，如需要在总数据中选取哪一年、哪一个生育期、哪一个处理的数据
                            只需要将所需要的生育期、处理编号数据即可返回所选择的数据
                        input:
                            strlist:  类型为list，列表中为一系列想要选取的处理编号如['ck1','2016',...]
                            location: 类型为list，列表中为一系列strlist中处理编号对应的列序列
                                example：index中处理编号为'ck1','ck2','ck3';生育期编号为'14','21','28';年份编号为'2016','2017'
                                        index中逐列的编号顺序为['**','**','处理','生育期','年份']
                                        1)选取2016年ck1和ck2的数据，strlist=['ck1','ck2','2016'],location=[2,4],即处理编
                                        号对应第三列，年份编号对应第五列
                                        2)选取2017年14生育期下ck2处理，strlist=['ck2','14','2017'],location=[2,3,4]
                                        3)选取2017年，strlist=['2017'],location=[4]
                        output:
                            order: 类型为array，整型，为所选取数据的索引
            >Decoration_corr:
                        装饰器,作为父类函数对之后的继承该类的函数起装饰作用，计算子类函数结果与生理指数的相关性，具体使用方法见子类
                        注意：父类中的装饰器不能对子类装饰，原因是没有实例父类，无法在子类找到装饰器函数，解决办法是在末尾将父类的
                        装饰器使用staticmethod()或classmethod()内置方法变为静态函数，可以直接使用类名.方法名()调用
                        https://stackoverflow.com/questions/3421337/accessing-a-decorator-in-a-parent-class-from-the-child-in-python
                        (StackOverflow yyds)
        栗子：
            >> import numpy as np
            >> from scipy.stats import pearsonr
            ##data_band;lai;index existence!!!
            >> w = base(data_band,lai)
            >> w.set_index(index)
            >> order = w.select_data_2(["ck1","2016"], [3,5])
            >> w.data_band[order,:] ## select all data like "2016-CK1""
        """
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


class red_side(base):

    """
    说明：定义求“三边”参数范围默认为680-760（可以使用lamda1和lamda2修改范围）、范围光谱数据求导、三边参数计算等函数
    属性：
        ata_band,lai,index与base相同
        lamda1:  类型为int，“三边”范围的开始波段，默认为680nm
        lamda2:  类型为int，“三边”范围的结束波段，默认为760nm
    内置函数：
        >derivative:
                    计算一个或多个光谱数据（指定范围）的一阶导数光谱，计算公式为 delta_R / delta_w
                input:
                    None
                output:
                    derivative_data: 类型为array，维度为2，返回一条或多条光谱数据的一阶导数计算结果
        >side:
                    使用Decoration_corr装饰，调用derivative函数，得出具体的“三边”参数，与生理指数的相关性，P值
                input:
                    None
                output:
                    side_data: 类型为array，维度为2，为计算的三边参数，逐列依次为maxWave,maxdata_band_d,area
                                maxWave为红边位置，范围导数最大值对应的波长
                                maxdata_band_d为红边斜率，范围导数最大值
                                area为红边面积，为范围内导数积分
                    R:          类型为array，维度为1，是各个三边参数与生理指数的相关性
                    p:          类型为array，维度为1，相关性显著检验
        >IG_red:
                    使用Decoration_corr装饰，另一种计算三边参数，与生理指数相关性，P值
                    计算方法：[J. R. MILLER, E. W. HARE & J. WU (1990) Quantitative characterization of 
                        the vegetation red edge reflectance 1. An inverted-Gaussian reflectance model, 
                        International Journal of Remote Sensing, 11:10, 1755-1773, DOI: 10.1080/01431169008955128]
                input:
                    None
                output:
                    IG_data: 类型为array，维度为2，为三边参数，逐列依次为 红边位置lamda0，红谷宽度sigma
                    R:       类型为array，维度为1，是各个三边参数与生理指数的相关性
                    p:       类型为array，维度为1，相关性显著检验
    栗子：
        >> import numpy as np
        >> from scipy.integrate import simps
        >> from scipy.stats import pearsonr
        ##data_band;lai;index existence!!!
        >> w = red_side(data_band,lai,index)
        >> data1,R1,P1 = w.side()
        >> data2,R2,P2 = w.IG_red()
    """

    def __init__(self,data_band,lai,lamda1=680,lamda2=760,index=None):
        super().__init__(data_band,lai,index=None)
        self.lamda1 = int(lamda1)
        self.lamda2 = int(lamda2)
        if self.data_band.ndim == 1:
            self.data_band = self.data_band.reshape((1,-1))
        else:
            self.data_band = self.data_band

    def derivative(self):
        derivative_data1 = self.data_band[:,0:(self.data_band.shape[1]-2)]
        derivative_data2 = self.data_band[:,2:self.data_band.shape[1]]
        derivative_data = (derivative_data2 - derivative_data1) / 2.0
        return derivative_data[:,self.lamda1-350:self.lamda2-350]

    @base.Decoration_corr
    def side(self):
        Wave = np.arange(350,1350)
        x = Wave[self.lamda1-350:self.lamda2-350]
        y = self.derivative()
        y = np.where(y<0,0,y)
        maxdata_band_d = np.max(y,axis=1)
        maxWave = np.argmax(y,axis=1) + self.lamda1
        area = simps(y,x,axis=1)
        side_data = np.concatenate((maxWave.reshape((-1,1)),maxdata_band_d.reshape((-1,1)),area.reshape((-1,1))),axis=1)
        return side_data

    @base.Decoration_corr
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

class vegetation_index(base):
    """
    说明：通过输入计算公式（变量定义具体规定）计算对应的植被指数，及与相应生理指数的相关性
    属性：
        data_band,lai,index与base相同
        veget_index_func:  类型为list，为所计算的植被指数对应的计算公式，具体命名格式为：
                                使用“R...”格式，...内容有以下5种形式
                                1)最常见的，直接是某个波长对应的反射率其参数为R+波长，如R810、R755等；
                                2)天依蓝波段，此时参数名称是固定的即Rblue，其返回蓝光范围（默认为435-450nm）反射率平均值，
                                    如果要修改范围在类实例化时将新范围传入即可，如vegetation_index(data_band,lai,blue=[1,2])，
                                    就是将默认的蓝光范围修改为1~2内，下同；
                                3)宝强绿波段，名称也为固定Rgreen（默认范围为492-577nm）可以传入green修改；
                                4)苏联红波段，名称固定Rred（默认范围为622-760nm）可以传入red修改；
                                5)近红外短波，名称固定Rnir（默认范围为780-1100nm）可以传入bir修改
                            使用以上格式的参数定义一系列计算植被指数的数学公式从而组成一个列表，如["R400+R600+1","R500-1"]
        names:              类型为list，为各个植被指数具体的名称，可以使用内置函数set_names进行设定，非必须
    内置函数：
        >set_names:
                    设定所计算植被指数的名称
                input:
                    names: 类型为list，为各个植被指数的名称
                output:
                    None
        >_find_symbol:
                    借助sympy库寻找一个植被指数计算公式中的未知变量，如计算公式为"2*R400+Rgreen-1"其变量为['R400','Rgreen']
                input:
                    None
                output:
                    sym_funcs: 类型为list，借助sympy将计算公式进行符号化，成为可运算的对象
                    forts:  类型为list，其中每个公式对应的变量均为一个dict，每个变量对应的值默认为1，如公式"2*R400+Rgreen-1"
                        返回的变量形式为{'R400':1,'Rgreen':1}，之后在运算时可修改对应的值进行运算
        >_find_data:
                    传入一个变量名称和光谱数据，根据变量格式计算其对应的值
                input：
                    string: 类型为str，为一个变量的名称
                    refl:  类型为array，维度为1，为一条光谱数据（无法使用矩阵计算只能forforfor了）
                output:
                    data: 类型为float，传入变量对应的数值
        >vegetation:
                    根据数据和公式计算相应的植被指数，并使用装饰器base.Decoration_corr计算结果与生理指数的相关性
                input:
                    None
                output:
                    vegetation_datas: 类型为array，维度为2，各个植被指数的计算结果
                    R:                类型为array，维度为1，各个植被指数与生理指数的相关性
                    P:                类型为array，维度为1，显著检验结果
    栗子：
        >> import numpy as np
        >> from scipy.stats import pearsonr
        >> from sympy import sympify,Symbol
        ##data_band;lai;index existence!!!
        >> funcs = ['R400+R500','Rred-R500+1']
        >> w =  vegetation_index(data_band,lai,funcs)
        >> data,R,P = w.vegetation()
    """
    extent = {'blue':[435,450],'green':[492,577],'red':[622,760],'nir':[780,1100]}
    names = []
    def __init__(self,data_band,lai,veget_index_func,index=None,**kwards):
        super().__init__(data_band, lai,index=None)
        if self.data_band.ndim == 1:
            self.data_band = self.data_band.reshape((1,-1))
        else:
            self.data_band = self.data_band
        self.veget_index_func = veget_index_func
        self.sym_funcs,self.forts = self._find_symbol()        
        for nume in kwards:
            vegetation_index.extent[nume][0] = int(kwards[nume][0])
            vegetation_index.extent[nume][1] = int(kwards[nume][1])

    def set_names(self,names):
        vegetation_index.names = names

    def _find_symbol(self):
        forts = []
        sym_funcs = []
        for str_func in self.veget_index_func:
            sym_func = sympify(str_func)
            sym_funcs.append(sym_func)    
            forts.append({x:1 for x in sym_func.atoms(Symbol)})
        return sym_funcs,forts
    
    def _find_data(self,string,refl):
        if string[1:] == 'blue':
            data = np.mean(refl[vegetation_index.extent['blue'][0]-350:vegetation_index.extent['blue'][1]-350])
        elif string[1:] == 'green':
            data = np.mean(refl[vegetation_index.extent['green'][0]-350:vegetation_index.extent['green'][1]-350])
        elif string[1:] == 'red':
            data = np.mean(refl[vegetation_index.extent['red'][0]-350:vegetation_index.extent['red'][1]-350])
        elif string[1:] == 'nir':
            data = np.mean(refl[vegetation_index.extent['nir'][0]-350:vegetation_index.extent['nir'][1]-350])
        else:
            data = refl[int(string[1:])-350]
        return data

    @base.Decoration_corr
    def vegetation(self):
        vegetation_datas = []
        for i in range(self.data_band.shape[0]):
            vegetation_data = []
            for sym_func,fort in zip(self.sym_funcs,self.forts):
                for variable in fort.keys():
                    #print(type(variable))
                    fort[variable] = self._find_data(str(variable),self.data_band[i])
                vegetation_data.append(sym_func.subs(fort).evalf())
            vegetation_datas.append(vegetation_data)
        vegetation_datas = np.array(vegetation_datas,dtype=np.float64)
        return vegetation_datas


