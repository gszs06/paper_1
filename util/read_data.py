#此部分主要定义了读写文件有关函数
import numpy as np

def read_physiological_index(index,fn):
    #读取生理指数函数
    """
    函数作用，传入光谱索引文件index和保存到本地的生理指数txt文件路径，返回一个与光谱索引一样长度的列表为每个光谱文件所对应的生理指数，需要指出的是生理
    指数的txt文件是特定格式的，分为两列，第一列为处理名称，名称组成为年份+距离开花的时间+处理名称（如‘201614ck1’），第二列为对应的生理指数，必须是这种
    格式否则会读写出错
    index: 为光谱索引文件
    fn: 为本地生理指数txt文件路径
    """
    data_txt = {}
    data = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        a = line.split('\t')
        data_txt[a[0]] = float(a[1])
    for i in range(index.shape[0]):
        name = index[i,4]+index[i,3]+index[i,2]
        data.append(data_txt[name])
    return data