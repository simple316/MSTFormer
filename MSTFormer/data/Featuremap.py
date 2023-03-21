import ipdb

from data.dataprocesslib import rad_distance, nmile2km, is_number
from numpy import *
import numpy as np
import os

import matplotlib.pyplot as plt

# ————————————————————————————————————————————参数设置————————————————————————————————————————————
SideLen_grid_input = 0.25  # lon,lat 一个格子多少公里
SAVE_PATH = str('data/')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
MaxVel = nmile2km(5)  # 海里转千米，km/h   #1节=1海里/小时=1.852公里/小时 一般一小时30~40海里
#72个点，71个时间间隔71*3=219min=3.65h 19个点，两两之间的时间间隔为3.65/19=0.192h 0.193…*40=7.72
# MaxVel = nmile2km(35)  # 海里转千米，km/h   #1节=1海里/小时=1.852公里/小时
# -----------------end ----------------------------------------------------------------xjy

# 轨迹点范围
# lat：17～31
# lon：-98.5～-80

GaussProbSigma = 0.7  # [0.4,1]

# 定义输入特征尺寸，计算一个多少海里
MaxDis = [MaxVel * 0.8, MaxVel * 1]  # 保证不会超过网格
MapRadio_input = [0, 0]
MapRadio_input[0] = int(round(MaxDis[0] / SideLen_grid_input))  # x方向半径（经度方向）
MapRadio_input[1] = int(round(MaxDis[1] / SideLen_grid_input))  # y方向半径（纬度方向）

HEADING_PROB = 0.2  # 船首向的概率值
SHIP_PROB = 0.8  # 船上一位置概率值
SPEED_PROB = 0.02  # 船上一位置速度表示概率值，将乘船的速度
SPEED_LINE = 3  # 速度拖尾长度

Rearth = 6371.004  ##地球平均半径,单位km,常数
UnitJWDis = math.pi * Rearth / 180  # 常数


# ——————————————————————————————————生成研究区域内的地理数据——————————————————————————————————
def PointInAreas(point, areas):
    InAreas = -1
    for i in range(len(areas)):
        if point[0] > areas[i, 0] and point[0] < areas[i, 2] and point[1] > areas[i, 1] and point[1] < areas[i, 3]:
            InAreas = i + 1
            return InAreas
    return InAreas


# center_jw代表中心点经纬度,gridjw代表网格内某点经纬度，求网格坐标，返回gridxy代表在这个局部矩阵中的坐标位置，中心点位置的坐标为（0，0），注意经度方向为x轴，纬度方向为y轴
def jw2grid(center_jw, gridjw, SideLen_grid):
    dis_x = rad_distance(gridjw[0], center_jw[1], center_jw[0], center_jw[1])  # 计算距离一定为正
    dis_y = rad_distance(center_jw[0], gridjw[1], center_jw[0], center_jw[1])
    gridx = np.sign(gridjw[0] - center_jw[0]) * (dis_x / SideLen_grid)
    gridy = np.sign(gridjw[1] - center_jw[1]) * (dis_y / SideLen_grid)
    # print('center:', center_jw, 'current', gridjw)
    # print(gridx, gridy)
    y = -round(gridy)
    x = round(gridx)
    if abs(y) > (MapRadio_input[0] - 3):
        if y > 0:

            y = MapRadio_input[0] - 3
        else:
            y = -(MapRadio_input[0] - 3)
    if abs(x) > (MapRadio_input[1] - 3):
        if x > 0:
            x = MapRadio_input[1] - 3
        else:
            x = -(MapRadio_input[1] - 3)
    return [y, x]  # 第一个坐标是y


# 其中gridxy代表在这个局部矩阵中的坐标位置，中心点位置的坐标为（0，0） gridxy分别为x轴（经度）和y轴（纬度）坐标
def grid2jw(center_jw, gridxy, SideLen_grid):
    gridw = gridxy[1] * SideLen_grid[1] / UnitJWDis + center_jw[1]
    gridj = gridxy[0] * SideLen_grid[0] / \
            (UnitJWDis * cos(gridw * np.pi / 180)) + center_jw[0]
    return [gridj, gridw]


def draw_map(map, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(map)  # ,cmap='Greys'
    # ax.invert_yaxis()
    plt.savefig('./img/prob_map/' + str(i) + '.jpg')


# ———————————————————————————————————————定义特征图类—————————————————————————————————————————
class Point:
    """
    表示一个点
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self,
               other):  # self是第一个参数。在python里是指“实例”本身。就是自己。other是第二个参数，它代表另一个class Point的实例。__eq__是一个重载等号的函数。意思是将两个class Point实例“=="，相当于判断两个实例中的x和y是否分别相等。
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __str__(
            self):  # 要把一个类的实例变成str。当你用print打印输出的时候，Python会调用它的str方法（p=Point(1,2) print p）。在我们编写一个新的Python类的时候，总是在最开始位置写一个初始化方法__init__，以便初始化对象，然后会写一个__str__方法，方面我们调试程序。
        return "x:" + str(self.x) + ",y:" + str(self.y)


def BresenhamDrawLine(img, start, end):
    #    1、输入线段的两个端点坐标和画线颜色：x1，y1，x2，y2，color；
    # 　　2、设置象素坐标初值：x = x1，y = y1；
    # 　　3、设置初始误差判别值：p = 2·Δy - Δx；
    # 　　4、分别计算：Δx = x2 - x1、Δy = y2 - y1；
    # 　　5、循环实现直线的生成：
    # 　　　for (x=x1；x <= x2；x++)
    #     　　　{putpixel(x, y, color) ；
    #     　　　　if (p >= 0)
    #     　　　　　{y=y+1；
    #     　　　　　　p=p+2·(Δy-Δx)；
    #     　　　　　}
    #     　　　　 else
    #     　　　　　{p=p+2·Δy；
    #     　　　　　}
    #     　　　}
    k = float(end[1] - start[1]) / (end[0] - start[0])
    dy = end[1] - start[1]
    dx = end[0] - start[0]
    x = start[0]
    y = start[1]
    if k < 1:
        p = 2 * dy - dx
        for x in range(start[0], end[0]):
            img[x][y] = 255
            if p >= 0:
                y += 1
                p += 2 * (dy - dx)
            else:
                p += 2 * dy
    else:
        p = 2 * dx - dy
        for y in range(start[1], end[1]):
            img[x][y] = 255
            if p >= 0:
                x += 1
                p += 2 * (dx - dy)
            else:
                p += 2 * dx


def WUAADrawLine(img, start, end, num):
    k = float(end[1] - start[1]) / (end[0] - start[0])
    if k < 1:
        y = start[1]
        for x in range(start[0], end[0]):
            off = y - int(y)
            img[x][int(y)] = img[x][int(y)] + (1 - off) * num
            img[x][int(y) + 1] = img[x][int(y) + 1] + off * num
            y += k
    else:
        k = float(end[0] - start[0]) / (end[1] - start[1])
        x = start[0]
        for y in range(start[1], end[1]):
            off = x - int(x)
            img[int(x)][y] = img[int(x)][y] + (1 - off) * num
            img[int(x) + 1][y] = img[int(x) + 1][y] + off * num
            x += k
    return img


class Map2D_current:  # 以前一时刻为中心，描述当前时刻相对前一时刻的位置（正方形）
    # center_jw前时刻经纬度，作为中心点；gridjw：下时刻经纬度；radio特征图的半径；SideLen_grid：特征图的网格边长
    def __init__(self, center_jw, gridjw, radio, SideLen_grid):
        self.center_jw = center_jw
        self.SideLen_grid = SideLen_grid
        self.gridxy = jw2grid(center_jw, gridjw, self.SideLen_grid)
        self.radio = radio  # [x,y],x和y方向两个数 中间点位置
        self.side_x = 2 * radio[0] + 1  # x方向半径（经度）
        self.side_y = 2 * radio[1] + 1  # y方向半径（纬度）

    def get_neighbors(self, px, py):
        for delta_x in [-2, -1, 0, 1, 2]:
            for delta_y in [-2, -1, 0, 1, 2]:
                dist = np.abs(delta_x) + np.abs(delta_y)
                if dist <= 2:
                    yield dist, int(px + delta_x), int(py - delta_y)

    def label_speed_heading(self, Sigma=0.5):
        pos_probabilty = np.zeros([self.side_x, self.side_y])
        center = np.array(self.radio)
        current = center + np.array(self.gridxy)
        pos_probabilty[int(center[0]), int(center[1])] = SHIP_PROB
        # print(center,current)
        for dist, px, py in self.get_neighbors(current[0], current[1]):
            if dist == 0:
                # print(px,py)
                pos_probabilty[px, py] = 0.50002
            elif dist == 1:
                pos_probabilty[px, py] = 0.114675
            elif dist == 2:
                pos_probabilty[px, py] = 0.00516

                # if self.SideLen_grid < 10:  # 按照边长6km计算
                #     if dis == 0:
                #         gause_prob = 0.50002
                #     elif dis == 1:
                #         gause_prob = 0.114675
                #     elif dis == 2:
                #         gause_prob = 0.00516
                #     else:
                #         gause_prob = 0
                #
                #     # 按照卫星拍摄误差分布修改
                #     # if dis==0:
                #     #    gause_prob = 0.34204793
                #     # elif dis==1:
                #     #    gause_prob = 0.069444444
                #     # elif dis==2:
                #     #    gause_prob = 0.017565359
                #     # elif dis==3:
                #     #    gause_prob = 0.008169935
                #     # elif dis==4:
                #     #    gause_prob = 0.004016885
                #     # elif dis==5:
                #     #    gause_prob = 0.002559913
                #     # elif dis==6:
                #     #    gause_prob = 0.001089325
                #     # else: gause_prob =0
                #
                # elif 10 <= self.SideLen_grid < 20:  # 按照边长12计算
                #     # if dis==0:
                #     #    gause_prob = 0.8262
                #     # elif dis==1:
                #     #    gause_prob = 0.04345
                #     # else:
                #     #    gause_prob =0
                #
                #     # 按照卫星拍摄误差分布修改
                #     if dis == 0:
                #         gause_prob = 0.471
                #     elif dis == 1:
                #         gause_prob = 0.0735
                #     elif dis == 2:
                #         gause_prob = 0.029375
                #     else:
                #         gause_prob = 0
                #
                #     # if dis==0:
                #     #    gause_prob = 0.471
                #     # elif dis==1:
                #     #    gause_prob = 0.0735
                #     # elif dis==2:
                #     #    gause_prob = 0.01275
                #     # elif dis==3:
                #     #    gause_prob = 0.00458
                #     # elif dis==4:
                #     #    gause_prob = 0.00244
                #     # elif dis==5:
                #     #    gause_prob = 0.00195
                #     # else:
                #     #    gause_prob =0
                # else:  # 按照边长20计算
                #     if dis == 0:
                #         gause_prob = 0.9768
                #     elif dis == 1:
                #         gause_prob = 0.0058
                #     else:
                #         gause_prob = 0

        # 针对船首向特征绘制概率图
        theta = self.center_jw[4] * math.pi / 180.0
        delta_x = int(np.around(math.sin(theta)))
        delta_y = int(np.around(math.cos(theta)))
        # print(self.center_jw[4], delta_x, delta_y)
        # print(center[0] + delta_x,center[1] + delta_y)
        pos_probabilty[center[0] - delta_y,center[1] + delta_x] = pos_probabilty[center[0] - delta_y,
                                                                       center[1] + delta_x] + HEADING_PROB

        # pos_probabilty[center_row][center_col] = pos_probabilty[center_row][center_col] + SHIP_PROB
        # if self.center_jw[4] == 0: pos_probabilty[center_row][center_col] = pos_probabilty[center_row][
        #                                                                         center_col] + HEADING_PROB
        # ship_heading = round(self.center_jw[4] / 45)
        # if ship_heading == 0:
        #     pos_probabilty[center_row][center_col + 1] = pos_probabilty[center_row][center_col + 1] + HEADING_PROB
        # elif ship_heading == -1:
        #     pos_probabilty[center_row + 1][center_col + 1] = pos_probabilty[center_row + 1][
        #                                                          center_col + 1] + HEADING_PROB
        # elif ship_heading == 1:
        #     pos_probabilty[center_row - 1][center_col + 1] = pos_probabilty[center_row - 1][
        #                                                          center_col + 1] + HEADING_PROB
        # elif ship_heading == 2:
        #     pos_probabilty[center_row - 1][center_col] = pos_probabilty[center_row - 1][center_col] + HEADING_PROB
        # elif ship_heading == -2:
        #     pos_probabilty[center_row + 1][center_col] = pos_probabilty[center_row + 1][center_col] + HEADING_PROB
        # elif ship_heading == -3:
        #     pos_probabilty[center_row + 1][center_col - 1] = pos_probabilty[center_row + 1][
        #                                                          center_col - 1] + HEADING_PROB
        # elif ship_heading == 3:
        #     pos_probabilty[center_row - 1][center_col - 1] = pos_probabilty[center_row - 1][
        #                                                          center_col - 1] + HEADING_PROB
        # else:
        #     pos_probabilty[center_row][center_col - 1] = pos_probabilty[center_row][center_col - 1] + HEADING_PROB
        # 针对船上一位置速度方向绘制位置

        # 针对船上一位置速度方向绘制位置
        speedprob = self.center_jw[2] * SPEED_PROB
        theta = self.center_jw[3] * math.pi / 180.0
        for i in range(SPEED_LINE):
            p = i + 1
            delta_x = int(np.around(p * math.sin(theta)))
            delta_y = int(np.around(p * math.cos(theta)))
            # print(delta_x, delta_y)
            pos_probabilty[center[0] + delta_y, center[1] - delta_x] = pos_probabilty[
                                                                           center[0] + delta_y, center[
                                                                               1] - delta_x] + speedprob
            # pos_probabilty = WUAADrawLine(pos_probabilty, center, end, speedprob)

            # center_row=current[0]
            # center_col=current[1]
            # speedprob = self.center_jw[2] / 20 * SPEED_PROB
            # cog = self.center_jw[3]

            # if abs(cog) == 90:  # 向上行驶
            #     for i in range(1, SPEED_LINE + 1):
            #         delta_y = int((-1) * (cog / abs(cog)) * i)  # 确定符号(方向问题,如果船向下行驶角度就为负值，将速
            #         # 度表示为尾巴时，是在上面，应为+)
            #         pos_probabilty[center_row][center_col + delta_y] = pos_probabilty[center_row][
            #                                                                center_col + delta_y] + speedprob
            # # elif self.center_jw[3]==-90: #向下行驶
            # #     for i in range(1, len(SPEED_LINE) + 1):
            # #         pos_probabilty[center_row, center_col+i]=pos_probabilty[center_row, center_col+i]+speedprob
            # else:
            #     tan_cog = tan(math.radians(cog))
            #     for j in range(1, SPEED_LINE + 1):
            #         if tan_cog > 1 or tan_cog < -1:  # 说明速度是竖着的，应该以变换行，确定列
            #             delta_col = round(j / tan_cog)
            #             if cog > 0:
            #                 delta_row = -1 * j
            #             else:
            #                 delta_row = 1 * j  # 确定符号(方向问题)
            #         else:
            #             delta_row = round(tan_cog * j)
            #             if cog > -90 and cog < 90:
            #                 delta_col = -1 * j
            #             else:
            #                 delta_col = 1 * j  # 确定符号(方向问题)
            #         pos_probabilty[center_row + delta_row][center_col + delta_col] = pos_probabilty[
            #                                                                              center_row + delta_row][
            #                                                                              center_col + delta_col] + speedprob

            # np.savetxt("./img/map.csv", pos_probabilty, fmt='%.2f', delimiter=",")
        return pos_probabilty


def map_data(traj):
    for i in range(len(traj)):
        traj[i][2] = nmile2km(traj[i][2])
    probmap = []
    for i in range(1, len(traj)):
        map_pos = Map2D_current(
            traj[i - 1, :], traj[i, :], MapRadio_input, SideLen_grid_input).label_speed_heading()
        # draw_map(map_pos, i)
        probmap.append(map_pos)
    return np.array(probmap)
