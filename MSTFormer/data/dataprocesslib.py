from math import radians, cos, sin, tan, asin, sqrt, atan, fabs
import time
import math
import matplotlib.pyplot as plt
import os
import sys
# import xlrd
import re

FilePath = os.path.dirname(os.path.realpath(__file__))
syspath = r"src"
sys.path.append(syspath)
import numpy as np

Rearth = 6371.004  ##地球平均半径,单位km,常数
UnitJWDis = math.pi * Rearth / 180  # 常数


##———————————————————评估网络模型——————————————————————
def grid2jw(center_jw, gridxy, SideLen_grid):  # 其中gridxy代表在这个局部矩阵中的坐标位置，中心点位置的坐标为（0，0） gridxy分别为x轴（经度）和y轴（纬度）坐标
    gridw = gridxy[1] * SideLen_grid / UnitJWDis + center_jw[1]
    gridj = gridxy[0] * SideLen_grid / (UnitJWDis * np.cos(gridw * np.pi / 180)) + center_jw[0]
    return [gridj, gridw]


def probmap2jwmap(probmap, center_j, center_w, SideLen_grid_output):  # 生成概率分布网格图对应的经纬网格坐标
    jmap = np.ones(probmap.shape)
    wmap = np.ones(probmap.shape)
    for index_x in range(probmap.shape[1]):  # probmap的列，表示经度方向
        for index_y in range(probmap.shape[0]):  # probmap的行，表示纬度方向
            grid_x = index_x - int(probmap.shape[1] / 2)  # x轴（经度）方向的网格坐标
            grid_y = int(probmap.shape[0] / 2) - index_y  # y轴（纬度）方向的网格坐标
            [gridj, gridw] = grid2jw([center_j, center_w], [grid_x, grid_y], SideLen_grid_output)
            jmap[index_y, index_x] = gridj
            wmap[index_y, index_x] = gridw
    return jmap, wmap


def probmap2jw(probmap, center_j, center_w, SideLen_grid_output,
               indication_size):  ##根据预测的概率分布图，选取目标指示范围内概率和最大的区域，给出经纬度位置
    grid_size = int(np.ceil(indication_size / SideLen_grid_output - int((indication_size / SideLen_grid_output) % 2)))
    regionprob = []
    for i in range(probmap.shape[0] - grid_size):
        for j in range(probmap.shape[1] - grid_size):
            probsum = np.sum(probmap[i:i + grid_size, j:j + grid_size])
            regionprob.append(np.array([probsum, i, j]))
    regionprob = np.array(regionprob)

    index = np.where(regionprob[:, 0] == np.max(regionprob[:, 0]))[0][0]
    index_sum = regionprob[index, 0]
    index_x = regionprob[index, 2] + int(grid_size / 2)  # 最大概率和区域的中心点在图中的位置,左上角为（0,0）
    index_y = regionprob[index, 1] + int(grid_size / 2)
    grid_x = index_x - int(probmap.shape[1] / 2)  # x轴（经度）方向的网格坐标
    grid_y = int(probmap.shape[0] / 2) - index_y  # y轴（纬度）方向的网格坐标
    bestjw = grid2jw([center_j, center_w], [grid_x, grid_y], SideLen_grid_output)
    bestprob_sum = index_sum
    return bestjw, bestprob_sum


# ——————————————————————————————————————————————————————————————————————————
'''trans_y(函数说明)
输入参数：
prob_predict表示预测模型输出的概率图，array格式，shape为m*OutputNode,m为预测样本集数量，OutputNode为网络节点输出个数，为概率图的边长的平方
center_jw表示概率图的中心点的经纬度坐标，即为当前时刻的经纬度，array格式，shape为m*2,2维表示中心点经度、中心点纬度
SideLen_grid_output表示概率图中一个网格的长度
indication_size表示目标指示范围的边长，如预测2小时indication_size为50km
输出参数：
np.array(jw_predict)：表示预测结果的最佳经纬度位置，根据目标指示范围内的最大概率和区域的中心点计算得出，array格式，shape为m*3，其中3维分别是最佳经度、最佳纬度、目标指示范围内的最大概率和。
np.array(probmap_set)：表示预测结果的经度图、纬度图、概率图，array格式，shape为m*3*a*a，其中3维分别是经度、纬度、概率，a*a表示网格图
prob_set：对probmap_set进行简化，仅输出结果概率值结果不为0的位置及概率，list格式，长度为m,其中每个list的尺寸为4*n,其中4维表示经度、纬度、概率、归一化概率，n为probmap中概率值不为0的个数。
'''


# ———————————————————————————————————————————————————————————————————————————
def trans_y(prob_predict, center_jw, SideLen_grid_output,
            indication_size):  # 将预测输出的prob分布转化为实际jw坐标,center_jw表示目标当前位置，即概率图中心点的jw坐标
    probmap_set = []
    jw_predict = []
    prob_set = []
    prob_predict = np.around(prob_predict, decimals=3, out=None)  # 将预测概率值小于0.00001的概率值忽略为0
    for i in range(len(prob_predict)):  # 第i个预测样本
        prob_map = prob_predict[i]
        prob_map = prob_map.reshape([int(np.sqrt(len(prob_map))), int(np.sqrt(len(prob_map)))])
        jmap, wmap = probmap2jwmap(prob_map, center_jw[i, 0], center_jw[i, 1], SideLen_grid_output)
        bestjw, bestprob_sum = probmap2jw(prob_map, center_jw[i, 0], center_jw[i, 1], SideLen_grid_output,
                                          indication_size)
        probmap_set.append(np.array(np.stack([jmap, wmap, prob_map])))
        jw_predict.append(np.array([bestjw[0], bestjw[1], bestprob_sum]))

        Index0, Index1 = np.nonzero(prob_map)
        prob_set.append(np.array(np.stack([jmap[Index0, Index1], wmap[Index0, Index1], prob_map[Index0, Index1],
                                           prob_map[Index0, Index1] / np.sum(prob_map[Index0, Index1])])))

    return np.array(jw_predict), np.array(probmap_set), prob_set


def bessel_2dot_fan(L10, B10, L20, B20):  # 输入为两点经度，纬度（小数制） 计算两点间方位角；不能处理矩阵
    if (L10 == L20 and B10 == B20):
        return 0, 0, 0
    e = 0.081813334016931499
    pi = 4 * atan(1)
    B1 = B10 * pi / 180
    L1 = L10 * pi / 180
    B2 = B20 * pi / 180
    L2 = L20 * pi / 180
    W1 = sqrt(1 - e * e * sin(B1) * sin(B1))
    W2 = sqrt(1 - e * e * sin(B2) * sin(B2))
    sinu1 = sin(B1) * sqrt(1 - e * e) / W1
    sinu2 = sin(B2) * sqrt(1 - e * e) / W2
    cosu1 = cos(B1) / W1
    cosu2 = cos(B2) / W2
    L = L2 - L1
    a1 = sinu1 * sinu2
    a2 = cosu1 * cosu2
    b1 = cosu1 * sinu2
    b2 = sinu1 * cosu2
    delta0 = 0
    lamda = L + delta0
    p = cosu2 * sin(lamda)
    q = b1 - b2 * cos(lamda) + 1e-5
    # print(L10,B10,L20,B20,p,q)
    A1 = atan(p / q)
    if p > 0:
        if q > 0:
            A1 = fabs(A1)
        else:
            A1 = pi - fabs(A1)
    else:
        if q > 0:
            A1 = 2 * pi - fabs(A1)
        else:
            A1 = pi + fabs(A1)
    sins = p * sin(A1) + q * cos(A1)  # 计算sigma的正弦值
    coss = a1 + a2 * cos(lamda)  # 计算sigma的余弦值
    sigma = atan(sins / coss)
    if coss > 0:
        sigma = fabs(sigma)
    else:
        sigma = pi - fabs(sigma)
    sinA0 = cosu1 * sin(A1)
    x = 2 * a1 - (1 - sinA0 * sinA0) * cos(sigma)
    afa = (33523299 - (28189 - 70 * (1 - sinA0 * sinA0)) * (1 - sinA0 * sinA0)) * 1.0e-10
    beta = (28189 - 94 * (1 - sinA0 * sinA0)) * 1.0e-10
    delta = (afa * sigma - beta * x * sin(sigma)) * sinA0
    lamda = L + delta
    while fabs(delta - delta0) > 4.8e-10:
        delta0 = delta
        p = cosu2 * sin(lamda)
        q = b1 - b2 * cos(lamda)
        A1 = atan(p / q)
        if p > 0:
            if q > 0:
                A1 = fabs(A1)
            else:
                A1 = pi - fabs(A1)
        else:
            if q > 0:
                A1 = 2 * pi - fabs(A1)
            else:
                A1 = pi + fabs(A1)
        sins = p * sin(A1) + q * cos(A1)  # 计算sigma的正弦值
        coss = a1 + a2 * cos(lamda)  # 计算sigma的余弦值
        sigma = atan(sins / coss)
        if coss > 0:
            sigma = fabs(sigma)
        else:
            sigma = pi - fabs(sigma)
        sinA0 = cosu1 * sin(A1)
        x = 2 * a1 - (1 - sinA0 * sinA0) * cos(sigma)
        afa = (33523299 - (28189 - 70 * (1 - sinA0 * sinA0)) * (1 - sinA0 * sinA0)) * 1.0e-10
        beta = (28189 - 94 * (1 - sinA0 * sinA0)) * 1.0e-10
        delta = (afa * sigma - beta * x * sin(sigma)) * sinA0
        lamda = L + delta

    A = 6356863.020 + (10708.949 - 13.474 * (1 - sinA0 * sinA0)) * (1 - sinA0 * sinA0)
    B = 10708.938 - 17.956 * (1 - sinA0 * sinA0)
    C = 4.487
    y = ((1 - sinA0 * sinA0) * (1 - sinA0 * sinA0) - 2 * x * x) * cos(sigma)
    S = A * sigma + (B * x + C * y) * sin(sigma)
    A2 = atan((cosu1 * sin(lamda)) / (b1 * cos(lamda) - b2) + 1e-5)
    if sin(A1) > 0:
        if tan(A2) > 0:
            A2 = pi + fabs(A2)
        else:
            A2 = 2 * pi - fabs(A2)
    else:
        if tan(A2) > 0:
            A2 = fabs(A2)
        else:
            A2 = pi - fabs(A2)
    A10 = A1 * 180 / pi  # 转化为角度
    A20 = A2 * 180 / pi  # 转化为角度
    S0 = S / 1000  # 单位为千米
    return S0, A10, A20  # 小数制 方向角（两个相对方向角） A10 为（L10,B10)到（L20,B20)的方位角，而A20为（L20,B20)到（L10,B10)的方位角


def rad_distance(lon1, lat1, lon2, lat2, Circle=True):  # 可以处理矩阵
    if Circle:
        ####---------- 地球近似为圆球体 ---------######
        Rearth = 6371.004  ##地球平均半径,单位km
        ang2rad = np.pi / 180
        tmp = np.sin(lat2 * ang2rad) * np.sin(lat1 * ang2rad) + np.cos(lat2 * ang2rad) * np.cos(
            lat1 * ang2rad) * np.cos(lon2 * ang2rad - lon1 * ang2rad)
        tmp = np.minimum(tmp, 1.0)
        tmp = np.maximum(tmp, -1.0)
        dis = Rearth * np.arccos(tmp)

    else:
        ####---------- 白塞尔大地距离---------######
        dis = []
        if (isinstance(lon1, float)):  ##判断变量类型是否是float型，这里主要是为了判断是数值还是numpy
            if (isinstance(lon2, float)):
                dis, _, _ = bessel_2dot_fan(lon1, lat1, lon2, lat2)
            else:
                for i in range(len(lon2)):
                    # print(lon1,lat1,lon2[i],lat2[i])
                    disi, _, _ = bessel_2dot_fan(lon1, lat1, lon2[i], lat2[i])
                    dis.append(disi)
        else:
            if (isinstance(lon2, float)):
                for i in range(len(lon1)):
                    disi, _, _ = bessel_2dot_fan(lon1[i], lat1[i], lon2, lat2)
                    dis.append(disi)
            else:
                for i in range(len(lon1)):
                    disi, _, _ = bessel_2dot_fan(lon1[i], lat1[i], lon2[i], lat2[i])
                    dis.append(disi)

    return np.array(dis)


def bessel_2dot_np(lon1, lat1, lon2, lat2):  # 可以处理矩阵
    ####---------- 白塞尔大地距离---------######
    dis, ang1, ang2 = [], [], []

    if (isinstance(lon1, float)):  ##判断变量类型是否是float型，这里主要是为了判断是数值还是numpy
        if (isinstance(lon2, float)):
            dis, ang1, ang2 = bessel_2dot_fan(lon1, lat1, lon2, lat2)
        else:
            for i in range(len(lon2)):
                # print(lon1,lat1,lon2[i],lat2[i])
                disi, ang1i, ang2i = bessel_2dot_fan(lon1, lat1, lon2[i], lat2[i])
                dis.append(disi)
                ang1.append(ang1i)
                ang2.append(ang2i)

    else:
        if (isinstance(lon2, float)):
            for i in range(len(lon1)):
                disi, ang1i, ang2i = bessel_2dot_fan(lon1[i], lat1[i], lon2, lat2)
                dis.append(disi)
                ang1.append(ang1i)
                ang2.append(ang2i)
        else:
            for i in range(len(lon1)):
                disi, ang1i, ang2i = bessel_2dot_fan(lon1[i], lat1[i], lon2[i], lat2[i])
                dis.append(disi)
                ang1.append(ang1i)
                ang2.append(ang2i)

    return np.array(dis), np.array(ang1), np.array(ang2)


def visibilityrate_evaluat(test_ylabel, test_predict, fukuan, daikuan, SideLen_grid_output, PRINT=True):  # 评价指标：成功捕获率

    detx, _, _ = bessel_2dot_np(test_ylabel[:, 0], test_ylabel[:, 1], test_predict[:, 0],
                                test_ylabel[:, 1])  ##预测位置与实际位置在经度方向的距离误差,同一纬度
    dety, _, _ = bessel_2dot_np(test_ylabel[:, 0], test_ylabel[:, 1], test_ylabel[:, 0],
                                test_predict[:, 1])  ##预测位置与实际位置在纬度方向的距离误差，同一经度
    inreg1 = (detx < (fukuan / 2)) * (dety < (daikuan / 2))

    detx, _, _ = bessel_2dot_np(test_predict[:, 0], test_predict[:, 1], test_ylabel[:, 0],
                                test_predict[:, 1])  ##预测位置与实际位置在经度方向的距离误差,同一纬度
    dety, _, _ = bessel_2dot_np(test_predict[:, 0], test_predict[:, 1], test_predict[:, 0],
                                test_ylabel[:, 1])  ##预测位置与实际位置在纬度方向的距离误差，同一经度
    inreg2 = (detx < (fukuan / 2)) * (dety < (daikuan / 2))

    detj = fukuan / 2 / (
                111 * np.cos(test_ylabel[:, 1] * np.pi / 180))  ##幅宽为东西向，对应同一纬度下，相邻经度的距离，fukuan/2/相邻经度的距离=fukuan/2距离的经度差
    detw = daikuan / 2 / 111  ##带宽为南北向，对应同一经度下，相邻纬度的距离，daikuan/2/相邻纬度的距离=daikuan/2距离的纬度差
    inreg3 = (abs(test_ylabel[:, 0] - test_predict[:, 0]) < detj) * (abs(test_ylabel[:, 1] - test_predict[:, 1]) < detw)

    inreg = np.array([inreg1, inreg2, inreg3])  ##验证三种结果是否有差异，更倾向于inreg1
    sate_acc = np.mean(inreg, axis=1) * 100  # 转化为百分比*100

    if PRINT: print('幅宽-带宽-目标可见率%： ', fukuan, daikuan, sate_acc)
    return sate_acc[0], inreg[0]


def timeh2str(time_h, fmt="%Y-%m-%d %H:%M:%S"):
    tm = time.localtime(time_h * 3600)
    return time.strftime(fmt, tm)


def str2timeh(tstr, fmt="%Y-%m-%dT%H:%M:%S"):
    tm = time.strptime(tstr, fmt)
    return time.mktime(tm) / 3600

def str2timeh_list(tstr, fmt="%Y-%m-%dT%H:%M:%S"):
    tm=[]
    for it in tstr:
        tm_=time.strptime(it, fmt)
        tm.append(time.mktime(tm_) / 3600)
    return tm

def nmile2km(nmile):  # 海里转千米.把地球看作球体，1nmile近似等于赤道所在的圆中1′的圆心角所对的弧长，即nmile*60*360=2*pi*R=地球赤道周长
    km = nmile * 1.852  # km=nmile*2*np.pi*Rearth/(360*60)
    return km


def Repeat_Clean(orig_txy):  ##清洗重复时刻数据
    clear_txy = np.unique(orig_txy, axis=0)
    return clear_txy


def readPort(path):  # 读取港口表格历史数据，参数是表格路径
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows
    dataAll = []
    pos = []
    for i in range(nrows):
        pos.append(table.row_values(i)[2:])
        dataAll.append([table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2], table.row_values(i)[3]])
    pos = np.array(pos)
    return dataAll, pos  ##时间（单位：小时）-经度-纬度


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
