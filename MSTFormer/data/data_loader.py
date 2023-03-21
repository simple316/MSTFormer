import copy
import random
from concurrent.futures import process
import os
import numpy as np
import pandas as pd
import random
import ipdb
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from numpy_ext import rolling_apply
import geopy.distance

from scipy import interpolate
import log as L
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from data.Featuremap import map_data, draw_map
from data.dataprocesslib import str2timeh

import warnings

warnings.filterwarnings('ignore')
INTER_TIME = 3  # 多久一个点
INTER_INDEX = 96


def even_select(N, M):
    """
    (near) Evenly select M samples from N instances.
    Return a mask array of size M, where thhe True value inidicates that the correponding element where be selected.
    """
    if N == M:
        return np.ones(N, dtype=bool)
    if N < M:
        raise ValueError("Try to select {} samples from {} instancees".format(M, N))
    if M > N / 2:
        cut = np.zeros(N, dtype=int)
        q, r = divmod(N, N - M)
        indices = [q * i + min(i, r) for i in range(N - M)]
        cut[indices] = True
    else:
        cut = np.ones(N, dtype=int)
        q, r = divmod(N, M)
        indices = [q * i + min(i, r) for i in range(M)]
        cut[indices] = False

    return np.logical_not(cut)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, prob_use=False,
                 features='S', data_path='ALL', data_type='ALL',
                 target='OT', scale=True, inverse=True, timeenc=0, freq='h', cols=None,
                 save_processed=None, read_processed=None, logger=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.prob_use = prob_use
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_type
        self.read_processed = read_processed
        self.logger = logger
        self.flag = flag
        self.__read_data__()

    def getFlist(self, file_dir):
        for root, dirs, files in os.walk(file_dir):
            s = 'root_dir:' + str(root) + '\n' + 'total number of file:' + str(len(files)) + '\n' + 'files:' + str(
                files)
            print(s)
            # print('root_dir:', root)  # 当前路径
            # print('total number of file:', len(files))
            # print('files:', files)  # 文件名称，返回list类型
        return dirs

    def draw_data(self, data):
        sca_lat = list(data[0][:, 0])
        sca_lon = list(data[0][:, 1])
        for i in range(1, len(data)):
            sca_lat.extend(list(data[i][:, 0]))
            sca_lon.extend(list(data[i][:, 1]))
        plt.scatter(sca_lon, sca_lat)
        plt.savefig('./img/data.jpg')

    def compute_distance(self, lon, lat):
        coord1 = (lon[0], lat[0])
        coord2 = (lon[1], lat[1])
        return geopy.distance.geodesic(coord1, coord2).nm

    # 计算角度差，顺时针为+，逆时针为—
    def compute_CourseDiverse(self, course):
        course1 = course[0]
        course2 = course[1]
        courseDiverse = 180 - (180 - course2 + course1) % 360
        if courseDiverse > 100:
            courseDiverse = courseDiverse - 360
        return courseDiverse

    def __read_data__(self):
        s = 'data prepare:' + str(self.flag) + 'train: 0, val: 1, test: 2'
        print(s)
        # print('data prepare:', self.set_type, 'train: 0, val: 1, test: 2')
        self.scaler = StandardScaler()

        # if self.prob_use == True:
        #     data_map_all = []
        #     all_len = self.seq_len + self.pred_len + 1  # all_len指整个数据长度
        # else:
        #     all_len = self.seq_len + self.pred_len
        all_len = self.seq_len + self.pred_len + 1
        # 判断文件夹是否存在,数据不存在，先把三种类型的数据都保存下来
        if not os.path.exists(self.read_processed):
            if self.data_path != 'ALL':
                files = [self.data_path]
            else:
                import glob
                if self.data_type[0] == 'ALL':
                    files = []
                    root_files = glob.glob(self.root_path + "*")
                    for i in range(len(root_files)):
                        # files.extend(glob.glob(root_files[i] + '/*.npy'))
                        if len(glob.glob(root_files[i] + '/*.npy')) < 68:
                            files.extend(glob.glob(root_files[i] + '/*.npy'))
                        else:
                            files.extend(glob.glob(root_files[i] + '/*.npy')[:68])
                else:
                    files = []
                    root_files = glob.glob(self.root_path + "*")
                    for i in range(len(root_files)):
                        if root_files[i].split('\\')[-1] in self.data_type:
                            files.extend(glob.glob(root_files[i] + '/*.npy'))
                            # if len(glob.glob(root_files[i] + '/*.npy')) < 68:
                            #     files.extend(glob.glob(root_files[i] + '/*.npy'))
                            # else:
                            #     files.extend(glob.glob(root_files[i] + '/*.npy')[:68])
                    # for i in range(len(self.data_type)):
                    #     root_files=self.root_path+'/'+self.data_type[i]
                    #     files.extend(glob.glob(root_files + '/*.npy'))
                # files=self.getFlist(self.root_path)

            '''1。读取文件夹所有数据，剔除空值数据'''
            s = 'number of data files:' + str(len(files))
            print(s)
            # print('number of data files:', len(files))
            data_all = []
            # files = files[:3]
            for ipath in files:
                '''读取npy'''
                df_raw = np.load(ipath, allow_pickle=True)
                df_raw = pd.DataFrame(df_raw)
                del df_raw['index']
                groups = df_raw.groupby(df_raw['group'])
                for group in groups:
                    # map_data = group[1][group[1]['true'] == 1]
                    data = group[1][group[1]['true'] == 0]
                    data = data[1:]
                    data = data[::INTER_TIME]
                    data = data.reset_index(drop=True)

                    '''获取时序数据'''
                    start = 0
                    while start + all_len < len(data):
                        data_ = data[start:start + all_len]
                        data_all_ = []
                        # map_data一个数据包含两个list，第一个是范围在encoder内
                        # map_data1_ = map_data[map_data['timeh'] >= data_['timeh'].iloc[0]]
                        # map_data1_ = map_data1_[map_data1_['timeh'] <data_['timeh'].iloc[self.seq_len]]
                        # mapdata1 = map_data1_[['LON', 'LAT', 'SOG', 'COG', 'Heading']].values
                        # map_data2_ = map_data[map_data['timeh'] >= data_['timeh'].iloc[self.seq_len]]
                        # map_data2_ = map_data2_[map_data2_['timeh'] <= data_['timeh'].iloc[-1]]
                        # mapdata2 = map_data2_[['LON', 'LAT', 'SOG', 'COG', 'Heading']].values
                        # if len(mapdata1) >= MAP_X_NUM and len(mapdata2) >= MAP_Y_NUM:
                        # mapdata = data_[['LON', 'LAT', 'SOG', 'COG', 'Heading']].values
                        # mapdata1=mapdata[:self.seq_len+1]
                        # mapdata2=mapdata[self.seq_len+1:]
                        # mask_array1 = even_select(len(mapdata1), MAP_X_NUM)  # select 10 maps
                        # mask_array2 = even_select(len(mapdata2), MAP_Y_NUM)  # select 10 maps
                        # data_map_all
                        # data_all_.append([mapdata1,mapdata2])
                        # data_all_.append([mapdata1[mask_array1, :],mapdata2[mask_array2, :]])
                        # data_['BaseDateTime'] = pd.to_datetime(data_['BaseDateTime'])
                        # data_stamp_ = time_features(data_[['BaseDateTime']], timeenc=self.timeenc, freq=self.freq)
                        # data_stamp_all
                        data_all_.append(list(data_['BaseDateTime'][1:]))
                        data_orig = data_[['LON', 'LAT', 'SOG', 'COG', 'Heading']].values
                        # data_all_true
                        data_all_.append(data_orig)
                        window = 2
                        data_['COG'] = rolling_apply(self.compute_CourseDiverse, window, data_['COG'].values)
                        data_[['LON', 'LAT', 'SOG']] = data_[['LON', 'LAT', 'SOG']].diff()
                        data_ = data_[['LON', 'LAT', 'SOG', 'COG']].values
                        # data_all_x
                        data_all_.append(data_[1:, :])
                        data_all.append(data_all_)
                        start = start + INTER_INDEX

            #     '''读取CSV'''
            #     # print(ipath)
            #     # ipath='./data/ship/366872170.csv'
            #     df_raw = pd.read_csv(ipath,
            #                          usecols=['BaseDateTime', 'LON', 'LAT', 'SOG', 'COG', 'Heading'])
            #     df_raw = df_raw[['BaseDateTime', 'LON', 'LAT', 'SOG', 'COG', 'Heading']]
            #     df_raw = df_raw.drop_duplicates(subset=['LON'])
            #     df_raw = df_raw.drop_duplicates(subset=['LAT'])
            #     # 删除有空缺值的行
            #     # df_raw.dropna(axis=0,how='any')
            #     # df_raw.dropna(subset=['SOG', 'COG', 'Heading'], how='any', inplace=True)
            #     df_raw.drop(df_raw.index[(df_raw['Heading'] == 511.0)], inplace=True)
            #     df_raw.drop(df_raw.index[(df_raw['SOG'] == 0.0)], inplace=True)
            #     df_raw = df_raw.reset_index(drop=True)
            #
            #     if len(df_raw) > all_len:
            #         '''2。计算出两两数据间时间差值（找出大于1小时的），根据encoder列筛选数据'''
            #         '''
            #         df_raw.columns: ['date', ...(other features), target feature]
            #         '''
            #         # cols = list(df_raw.columns);
            #         if self.cols:
            #             cols = self.cols.copy()
            #             cols.remove(self.target)
            #         else:
            #             cols = list(df_raw.columns)
            #             cols.remove(self.target)
            #             cols.remove('BaseDateTime')
            #         df_raw = df_raw[['BaseDateTime'] + cols + [self.target]]
            #         df = df_raw
            #         df['BaseDateTime'] = pd.to_datetime(df_raw['BaseDateTime'])
            #
            #         # 清理异常速度
            #         window = 2
            #         # df['speed'] = rolling_apply(self.compute_distance, window, df['LON'].values, df['LAT'].values)
            #         lat_diff = df["LAT"].rolling(window=2)
            #         lon_diff = df["LON"].rolling(window=2)
            #         dists = []
            #         for lat, lon in zip(lat_diff, lon_diff):
            #             if len(lon) == 1:
            #                 dists.append(np.nan)
            #             else:
            #                 coord1 = (lat.iloc[0], lon.iloc[0])
            #                 coord2 = (lat.iloc[1], lon.iloc[1])
            #                 dist = geopy.distance.geodesic(coord1, coord2).km
            #                 dists.append(dist)
            #         df['timeh'] = df_raw['BaseDateTime'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S')).apply(str2timeh)
            #         df["speed"] = dists / df['timeh'].diff()
            #         df = df[df['speed'] < 20]
            #         del df["speed"]
            #         df = df.reset_index(drop=True)
            #
            #         df["hour"] = df['BaseDateTime'].apply(lambda x: x.hour)
            #         df["minute"] = df['BaseDateTime'].apply(lambda x: x.minute)
            #         df["minute"] = (df["minute"] / INTER_TIME).astype(int)
            #         df["day"] = df['BaseDateTime'].apply(lambda x: x.dayofyear)
            #         # 更改间隔hour，day都要改，按照初始值15分钟一个点，一小时4个，一天24*4
            #         df["hour"] = df["minute"] + df["hour"] * (60 / INTER_TIME) + df["day"] * 24 * (60 / INTER_TIME)
            #         df.drop_duplicates(subset=['hour'], keep='first', inplace=True)
            #         df = df.reset_index(drop=True)
            #         df['timediffer'] = df["hour"].diff()
            #         # df['timediffer'][0] = 3.0
            #         # df['timediffer'][-1] = 3.0
            #         del df["hour"]
            #         del df["day"]
            #         del df["minute"]
            #         # del df['timediffer']
            #         df_raw = df.copy(deep=True)
            #         del df_raw['timediffer']
            #         df_timeh = df_raw['timeh'].values
            #         del df_raw['timeh']
            #
            #         if self.features == 'M' or self.features == 'MS':
            #             cols_data = df_raw.columns[1:]
            #             df_data = df_raw[cols_data]
            #         elif self.features == 'S':
            #             df_data = df_raw[[self.target]]
            #
            #         part_index_ = df[df['timediffer'] > 2].index.tolist()
            #         part_index = []
            #
            #         '''3。数据归一化、数据拆分'''
            #         for i in range(1, len(part_index_)):
            #             if (part_index_[i] - part_index_[i - 1]) > all_len + 1:
            #                 part_index.append([part_index_[i - 1], part_index_[i]])
            #         if len(part_index) > 0:
            #             df_data = df_data.values
            #             df_stamp = df_raw[['BaseDateTime']]
            #             data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            #
            #             for i in range(len(part_index)):
            #                 # start = int(part_index[i][0] + 1)
            #                 start = part_index[i][0]
            #                 # if part_index[i]-start>self.seq_len:
            #                 end = start + all_len + 1
            #                 while (end < part_index[i][1]):
            #                     if len(data_all) == 0:
            #                         data_all = [df_data[start:end, :]]
            #                         stamp_all = [data_stamp[start:end]]
            #                         time_real = [df_timeh[start:end]]
            #                     else:
            #                         data_all.append(df_data[start:end, :])
            #                         stamp_all.append(data_stamp[start:end])
            #                         time_real.append(df_timeh[start:end])
            #                     '''此处更改间隔'''
            #                     start = start + 2
            #                     # start = start + all_len
            #                     end = start + all_len + 1
            #                 #     data_all = [df_data[start:part_index[i][1], :]]
            #                 #     stamp_all = [data_stamp[start:part_index[i][1]]]
            #                 #     # if self.prob_use == True:
            #                 #     #     # data_map_all = [df_data[(start - 1):part_index[i][1], :]]
            #                 #     #     data_map_all = [df_data[start:part_index[i][1], :]]
            #                 # else:
            #                 #     data_all.append(df_data[start:part_index[i][1], :])
            #                 #     stamp_all.append(data_stamp[start:part_index[i][1]])
            #                 #     # if self.prob_use == True:
            #                 #     #     # data_map_all.append(df_data[(start - 1):part_index[i][1], :])
            #                 #     #     data_map_all.append(df_data[start:part_index[i][1], :])
            # # self.draw_data(data_all)
            # random.shuffle(data_all_x)
            random.shuffle(data_all)
            num_train = int(round(len(data_all) * 0.7))
            num_test = num_train + int(round(len(data_all) * 0.2))
            num_vali = len(data_all) - num_train - num_test
            border1s = [0, num_train, num_test]
            border2s = [num_train, num_test, len(data_all)]

            '''重复保存，train，test，vali三种数据'''
            data_type = ['train', 'test', 'val']
            for i in range(len(data_type)):
                border1 = border1s[i]
                border2 = border2s[i]
                # data_all :[[sample data for map],[time code data],[original data(no heading)],[delta data]]
                data_all_delta = list(list(zip(*data_all[border1:border2]))[2]).copy()
                data_all_orig = list(list(zip(*data_all[border1:border2]))[1]).copy()
                data_stamp_all = list(list(zip(*data_all[border1:border2]))[0]).copy()
                # data_map_all = list(list(zip(*data_all[border1:border2]))[0]).copy()
                if self.read_processed is not None:
                    if not os.path.exists(self.read_processed):
                        os.makedirs(self.read_processed)
                    save_path = "{}-{}.pkl".format(self.read_processed, data_type[i])
                    s = "Save processed data to {}".format(save_path)
                    print(s)
                    torch.save({"data_all_delta": data_all_delta, "data_all_original": data_all_orig,
                                "data_stamp": data_stamp_all}, save_path)
        read_path = "{}-{}.pkl".format(self.read_processed, self.flag)
        s = "Read processed data from {}".format(read_path)
        print(s)
        processed_data = torch.load(read_path)
        data_all_delta = processed_data["data_all_delta"]
        data_all_orig = processed_data["data_all_original"]
        data_stamp_all = processed_data["data_stamp"]
        # data_map_all = processed_data["data_map"]       #old data
        if self.flag == 'train':
            train_data = data_all_delta
        else:
            read_path = "{}-{}.pkl".format(self.read_processed, 'train')
            train_data = torch.load(read_path)["data_all_delta"]

        ''''时序数据编码'''
        data_ = pd.DataFrame(np.array(data_stamp_all).reshape(-1, 1), columns=['BaseDateTime'])
        data_stamp_all = time_features(data_, timeenc=self.timeenc, freq=self.freq)
        data_stamp_all = data_stamp_all.reshape(-1, all_len - 1, data_stamp_all.shape[1])
        '''选择输入，输出'''
        data_all_delta = np.array(data_all_delta)[:, :, 2:]  # \delta x
        data_all_orig = np.array(data_all_orig)  # 真实轨迹
        # [真实轨迹(72),\delta x(24)],loss是针对轨迹差值，保留data_all_x，作为真值，因为后续归一化后x会改变
        # data_all_y = np.concatenate([data_all_y[:,:self.seq_len,:],data_all_x[:,self.seq_len:,:]], 1)
        train_data = np.array(train_data)[:, :, 2:]
        '''数据归一化'''
        if self.scale:
            train_data = train_data.reshape(-1, len(train_data[0][0]))
            self.scaler.fit(train_data)
            data_all_delta = self.scaler.transform(data_all_delta)
        else:
            data_all_delta = data_all_delta
        self.data_delta = data_all_delta  #这里只保存归一化数据的原因是，可以利用self.scaler.inverse_transform函数还原数据
        self.data_orig = data_all_orig
        self.data_stamp = data_stamp_all
        # self.data_map = data_map_all         #old data



    def __getitem__(self, index, true_lat=None):
        s_begin = 0  # index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = np.array(self.data_delta[index][s_begin:s_end])
        # atten_x=np.array(沿着指定的维度重复张量的元素[s_begin:s_end])
        # time_all = np.array(self.data_time[index][s_begin:r_end])
        if self.inverse:
            seq_y = np.array(np.concatenate([self.data_delta[index][r_begin:r_begin + self.label_len],
                                             self.scaler.inverse_transform(
                                                 self.data_delta[index][r_begin + self.label_len:r_end])],
                                            0))  # 连接decoder，预测点真值
        else:
            seq_y = np.array(self.data_delta[index][r_begin:r_end])
        # atten_y=np.array(self.data_orig[index][r_begin:r_begin + self.label_len])
        if self.prob_use == True:
            data_map = [self.data_orig[index][:self.seq_len + 1], self.data_orig[index][self.seq_len + 1:]]
            MAP_X_NUM = int(self.seq_len / 4 + 1)
            MAP_Y_NUM = int(self.pred_len / 4 + 1)
            mask_array1 = even_select(len(data_map[0]), MAP_X_NUM)  # select 10 maps
            mask_array2 = even_select(len(data_map[1]), MAP_Y_NUM)  # select 10 maps
            map_x = map_data(data_map[0][mask_array1, :])  # map_data计算图
            map_y = map_data(data_map[1][mask_array2, :])
            # map_x= map_data(data_map[0])
            # map_y= map_data(data_map[1])
            # map_x = map_data(self.data_map[index][0])
            # map_y = map_data(self.data_map[index][1])
        seq_x_mark = np.array(self.data_stamp[index][s_begin:s_end])
        seq_y_mark = np.array(self.data_stamp[index][r_begin:r_end])
        seq_orig = np.array(self.data_orig[index][:, :4])

        # # 插值预测结果
        # t = []
        # lon = []
        # lat = []
        # t0 = self.data_time[index][s_end - 1]
        # inter_t = INTER_TIME / 60
        # true_t = np.linspace(inter_t, (self.pred_len) * inter_t, self.pred_len).tolist()
        # for i in range(-1, self.pred_len + 1):
        #     t.append(self.data_time[index][s_end + i] - t0)
        #     lon.append(self.data_y[index][s_end + i][0])
        #     lat.append(self.data_y[index][s_end + i][1])
        # f_lon = interpolate.interp1d(t, lon, kind='linear')
        # f_lat = interpolate.interp1d(t, lat, kind='linear')
        # true_lon = np.array([f_lon(true_t)])
        # true_lon = true_lon.reshape(self.pred_len, 1)
        # true_lat = np.array([f_lat(true_t)])
        # true_lat = true_lat.reshape(self.pred_len, 1)
        # true = np.concatenate((true_lon, true_lat), 1)
        # s_begin = s_begin + self.seq_len + self.pred_len
        # while (s_begin < int(len(self.data_x[index]) / (self.seq_len + self.pred_len)) * (
        #         self.seq_len + self.pred_len)):
        #     s_end = s_begin + self.seq_len
        #     r_begin = s_end - self.label_len
        #     r_end = r_begin + self.label_len + self.pred_len
        #     seq_x = np.concatenate((seq_x, np.array([self.data_x[index][s_begin:s_end]])))
        #     if self.inverse:
        #         seq_y_ = np.concatenate([self.data_x[index][r_begin:r_begin + self.label_len],
        #                                  self.data_y[index][r_begin + self.label_len:r_end]], 0)
        #         seq_y = np.concatenate((seq_y, np.array([seq_y_])))
        #     else:
        #         seq_y = np.concatenate((seq_y, np.array([self.data_y[index][r_begin:r_end]])))
        #     if self.prob_use == True:
        #         map_x = np.concatenate((map_x, map_data(self.data_x[index][s_begin:s_end])))
        #         map_y = np.concatenate((map_y, map_data(self.data_y[index][r_begin:r_end])))
        #         # map_x = np.concatenate((map_x, map_data(self.data_map[index][s_begin:s_end])))
        #         # map_y = np.concatenate((map_y, map_data(self.data_map[index][r_begin:r_begin + self.label_len])))
        #     seq_x_mark = np.concatenate((seq_x_mark, np.array([self.data_stamp[index][s_begin:s_end]])))
        #     seq_y_mark = np.concatenate((seq_y_mark, np.array([self.data_stamp[index][r_begin:r_end]])))
        #     s_begin = s_begin + self.seq_len + self.pred_len
        if self.prob_use == True:
            return seq_x, seq_y, map_x, map_y, seq_x_mark, seq_y_mark, seq_orig
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_orig


    def __len__(self):
        # return len(self.data_x) - self.seq_len- self.pred_len + 1
        # len_=0
        # for i in range(len(self.data_x)):
        #     len_=len_+int(len(self.data_x[i])/(self.seq_len + self.pred_len))
        # return len_
        return len(self.data_delta)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

