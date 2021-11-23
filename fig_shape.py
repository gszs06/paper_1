import numpy as np
import PyEMD
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fft(eemd_data):
    fft_data = []
    for i in range(len(eemd_data)):
        a = np.fft.fft(eemd_data[i])
        fft_data.append(np.abs(a))
    fft_data  = np.array(fft_data)
    return fft_data

def find_max(fft_data):
    fftmax_index = []
    fftmax_data = []
    for i in range(fft_data.shape[0]):
        fftmax_index.append(np.argmax(fft_data[i,0:500])+1)
        fftmax_data.append(np.max(fft_data[i,0:500]))
    return fftmax_index,fftmax_data

def plot_eemd_fft(eemd_data,fft_data,fftmax_index,fftmax_data,path=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    Wave = np.arange(350,1350)
    fig = plt.figure(figsize=(18,24),dpi=200,constrained_layout=True)
    spc = fig.add_gridspec(ncols=9,nrows=8)
    for i in range(eemd_data.shape[0]):
        ax1 = fig.add_subplot(spc[i,0:4])
        ax1.plot(Wave,eemd_data[i,:],color='black')
        ax1.set_xticklabels([])
        ax1.set_ylabel("imf {:s}".format(str(i+1)),fontsize=20)
        if i == 0:
            plt.title("a:EEMD分解结果",fontsize=20)
        ax2 = fig.add_subplot(spc[i,4:8])
        ax2.plot(fft_data[i,:],color='black')
        ax2.set_xticklabels([])
        if i == 0:
            plt.title("b:对分解结果进行FFT变换",fontsize=20)
        ax3 = fig.add_subplot(spc[i,8])
        ax3.annotate("位置为:{:d}\n值为:{:.2f}".format(fftmax_index[i],fftmax_data[i]),
                    xy=(0.5,0.5),xycoords='data',
                    va="center", ha="center",bbox=dict(fc='w',ec='black'),
                    fontsize=30)
        ax3.set_axis_off()
        if i == 0:
            plt.title("c:变换最大值",fontsize=20)
        if i == eemd_data.shape[0]-1:
            ax1.set_xticks([350,550,750,950,1150,1350])
            ax1.set_xticklabels([350,550,750,950,1150,1350])
            ax1.set_xlabel("波长(nm)",fontsize=20)
            ax2.set_xticks([0,200,400,600,800,1000])
            ax2.set_xticklabels([0,200,400,600,800,1000])
    fig.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def find_point(nonoise_data):
    peak_points_1 = find_peaks(nonoise_data[:400])[0]
    peak_points_2 = find_peaks(nonoise_data[400:600])[0] + 400
    peak_points_3 = find_peaks(nonoise_data[600:])[0] + 600
    peak_point_200 = peak_points_1[np.argmax(nonoise_data[peak_points_1])]
    peak_point_400 = peak_points_2[0]
    peak_point_500 = peak_points_2[-1]
    peak_point_700 = peak_points_3[0]
    peak_point_900 = peak_points_3[-1]
    peak_point_200 = np.array(peak_point_200,dtype=np.int)
    peak_point_400 = np.array(peak_point_400,dtype=np.int)
    peak_point_500 = np.array(peak_point_500,dtype=np.int)
    peak_point_700 = np.array(peak_point_700,dtype=np.int)
    peak_point_900 = np.array(peak_point_900,dtype=np.int)
    peak_points = np.hstack((peak_point_200,peak_point_400,peak_point_500,peak_point_700,peak_point_900))
    valley_points = []
    valley_points.append(np.argmin(nonoise_data[peak_points[0]:peak_points[1]]) + peak_points[0])
    valley_points.append(np.argmin(nonoise_data[peak_points[2]:peak_points[3]]) + peak_points[2])
    valley_points.append(np.argmin(nonoise_data[peak_points[3]:peak_points[4]]) + peak_points[3])           
    valley_points = np.array(valley_points,dtype=np.int)
    return peak_points,valley_points

def line_inter(x,y):
    X = np.arange(x[0],x[1])
    #X = np.linspace(x[0],x[1],num=50)
    Y = (X-x[1])*(y[0]-y[1])/(x[0]-x[1]) + y[1]
    return X,Y

def make_line_data(nonoise_data,peak_points):
    _,line24_y = line_inter(peak_points[0:2],nonoise_data[peak_points[0:2]])
    _,line6_y = line_inter(peak_points[2:4],nonoise_data[peak_points[2:4]])
    _,line8_y = line_inter(peak_points[3:5],nonoise_data[peak_points[3:5]])
    line_data = nonoise_data.copy()
    line_data[peak_points[0]:peak_points[1]] = line24_y
    line_data[peak_points[2]:peak_points[3]] = line6_y
    line_data[peak_points[3]:peak_points[4]] = line8_y
    return line_data

def plot_noise(original_data,nonoise_data,path=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    Wave = np.arange(350,1350)
    fig = plt.figure(figsize=(10,4),dpi=200)
    ax1 = fig.add_subplot(121)
    ax1.plot(Wave,original_data,color='black')
    ax1.set_xticks([350,550,750,950,1150,1350])
    ax1.set_xlabel('波长(nm)',fontsize=14)
    ax1.set_ylabel('反射率',fontsize=14)
    plt.title("a:原始光谱曲线")

    ax2 = fig.add_subplot(122)
    ax2.plot(Wave,nonoise_data,color='black')
    ax2.set_xticks([350,550,750,950,1150,1350])
    ax2.set_xlabel('波长(nm)',fontsize=14)
    plt.title("b:使用EEMD去噪后光谱曲线")
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def plot_normway(line_data,nonoise_data,norm_data,points,path=None):
    points_y = nonoise_data[points]
    x = [np.arange(553,673),np.arange(673,788),
        np.arange(874,969),np.arange(969,1072),
        np.arange(1072,1192),np.arange(1192,1257)]
    y1 = [[1]*120,[1]*115,[1]*95,[1]*103,[1]*120,[1]*65]
    y2 = [norm_data[203:323],norm_data[323:438],
        norm_data[524:619],norm_data[619:722],
        norm_data[722:842],norm_data[842:907]]
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    Wave = np.arange(350,1350)
    fig = plt.figure(figsize=(10,4),dpi=200)
    ax1 = fig.add_subplot(121)
    ax1.plot(Wave,line_data,color='black',linestyle=':',label='Linear Envelope')
    ax1.scatter(points+350,points_y,c='black',s=15,label='Extreme Point')
    ax1.plot(Wave,nonoise_data,color='black',label='Spectral Curve')
    ax1.set_xticks([350,550,750,950,1150,1350])
    ax1.set_xlabel('Wavelength (nm)',fontsize=14)
    ax1.set_ylabel('Reflectance',fontsize=14)
    ax1.legend(edgecolor='w',fontsize=13,loc=0)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(Wave,norm_data,color='black',label='Normalized Spectral \nCurve')
    ax2.set_xticks([350,550,750,950,1150,1350])
    ax2.set_xlabel('Wavelength (nm)',fontsize=14)
    ax2.set_ylabel('Normalized Reflectance',fontsize=14)
    ax2.legend(edgecolor='w',fontsize=13,loc=0)
    # ax2.annotate('$D_{673}$',xy=(673,norm_data[673-350]),
    #             xycoords='data',xytext=(500,0.07),
    #             arrowprops=dict(arrowstyle="->"))
    ax2.annotate('D2',xy=(969,norm_data[969-350]),
                xycoords='data',xytext=(900,0.7),
                arrowprops=dict(arrowstyle="->"))
    ax2.annotate('D3',xy=(1192,norm_data[1192-350]),
                xycoords='data',xytext=(1150,0.6),
                arrowprops=dict(arrowstyle="->"))
    ax2.annotate('A1',xy=(600,0.6),xytext=(350,0.5),arrowprops=dict(arrowstyle="->"))
    ax2.annotate('A2',xy=(720,0.7),xytext=(750,0.4),arrowprops=dict(arrowstyle="->"))
    ax2.annotate('A3',xy=(940,0.95),xytext=(780,0.8),arrowprops=dict(arrowstyle="->"))
    ax2.annotate('A4',xy=(1000,0.93),xytext=(1050,0.65),arrowprops=dict(arrowstyle="->"))
    ax2.annotate('A5',xy=(1150,0.95),xytext=(1030,0.4),arrowprops=dict(arrowstyle="->"))
    ax2.annotate('A6',xy=(1200,0.9),xytext=(1250,0.7),arrowprops=dict(arrowstyle="->"))
    hatch = '//'
    for i in range(6):
        ax2.fill_between(x[i],y1[i],y2[i],hatch=hatch,facecolor='lightgray')
        if i%2 ==0:
            hatch = '\\\\'
        else:
            hatch = '//'
    
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()



np.set_printoptions(precision=3,suppress=True)
import os

data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")

running = 'python {}'.format(os.path.abspath("script/eemd.py"))
data_input = "277"
data_input = bytes(data_input,encoding='utf8')
# a = subprocess.run(args=running,input=data_input,capture_output=True)
eemd_data = np.load(os.path.abspath("F:/code/graduation_thesis_a/data/{:s}_eemd.npy".format(data_input.decode())))
# fft_data = section2_function.fft(eemd_data)
# fftmax_index,fftmax_data = section2_function.find_max(fft_data)
# section2_function.plot_eemd_fft(eemd_data,fft_data,fftmax_index,fftmax_data,path="F:/吟咏之间，吐纳珠玉之声/论文改2/2章/图片/1.png")

original_data = data_band[277,:]
nonoise_data = np.sum(eemd_data[4:],axis=0)
# section2_function.plot_noise(original_data,nonoise_data,path="F:/吟咏之间，吐纳珠玉之声/论文改2/2章/图片/2.png")

peak_points,valley_points = find_point(nonoise_data)
line_data = make_line_data(nonoise_data,peak_points)
norm_data = nonoise_data/line_data
norm_data = np.where(norm_data>1,1,norm_data)

points = np.hstack((peak_points,valley_points))
points = np.sort(points)
plot_normway(line_data,nonoise_data,norm_data,points,path="F:/paper/paper_1/吸收谷参数方法.png")

