import ipdb
import torch
from torch import nn
import math

TIMEINTER=3 #时间间隔，单位为min

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def distance(self, lon1, lat1, lon2, lat2): #结果为公里
        # radius = 6371  # km
        radius = self.get_earth_radius(lat1,lon1)  # km

        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        a = torch.sin(dlat / 2) * torch.sin(dlat / 2) + torch.cos(torch.deg2rad(lat1)) \
            * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2) * torch.sin(dlon / 2)
        c = 2 * torch.atan2(torch.sqrt(a + 1e-10), torch.sqrt(1 - a + 1e-10))
        d = radius * c
        return d

    def get_earth_radius(self, lat, lon):
        """get earth radius by WSG-84 """
        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)
        a = 6378137.0
        b = 6356752.3142
        an = a * a * torch.cos(lat)
        bn = b * b * torch.sin(lat)
        ad = a * torch.cos(lat)
        bd = b * torch.sin(lat)
        return torch.sqrt((an * an + bn * bn) / (ad * ad + bd * bd)+1e-10) /1000

    def sog_cog_position(self, lon1, lat1, sog, cog):
        """
        已知起点经纬度，使用距离与方位角求终点经纬度
        :param lat1: 已知纬度
        :param lon1: 已知经度
        :param azimuth: 已知方位角 °
        :param distance: 已知距离 km 根据速度计算距离
        :return: 终点经纬度
        """
        radius = self.get_earth_radius(lat1, lon1)
        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)
        cog = torch.deg2rad(cog)
        distance = sog * (TIMEINTER/60)
        distance = distance * 1.852  #海里转化为km
        distance = distance / radius   # radius of the earth in km
        lat2 = torch.asin(
            torch.sin(lat1) * torch.cos(distance) + torch.cos(lat1) * torch.sin(distance) * torch.cos(cog))
        lon2 = lon1 + torch.atan2(torch.sin(cog) *torch.sin(distance) * torch.cos(lat1),
                                 torch.cos(distance) - torch.sin(lat1) * torch.sin(lat2))
        lat2 = torch.rad2deg(lat2)
        lon2 = torch.rad2deg(lon2)
        position = torch.cat([lon2, lat2], 2)
        return position

    def Correction_course(self,data):
        for i in range(len(data)):
            if data[i][0][1] > 360:
                data[i][0][1] = data[i][0][1] - 360
            elif data[i][0][1] < 0:
                data[i][0][1] = data[i][0][1] + 360
        return data

    # #直接使用SOG和COG预测位置
    # def get_true_byDeltaSogCog(self,preds, trues):
    #     preds[:, 0:1, :] = self.Correction_course(trues[:, 0:1, 2:] + preds[:, 0:1, :])
    #     posi_preds = self.sog_cog_position(trues[:, 0:1, :1], trues[:, 0:1, 1:2],
    #                                   trues[:, 0:1, 2:3], trues[:, 0:1, 3:4])
    #     for i in range(1,len(preds[0])):
    #         preds[:, i:i + 1, :] = self.Correction_course(preds[:, i - 1:i, :] + preds[:, i:i + 1, :])
    #         posi_preds_ = self.sog_cog_position(posi_preds[:, i - 1:i, :1], posi_preds[:, i - 1:i, 1:2],
    #                                        preds[:, i - 1:i, :1], preds[:, i - 1:i, 1:])
    #         posi_preds = torch.cat([posi_preds, posi_preds_], 1)
    #     return posi_preds

    # 使用修正后的SOG和COG预测位置
    def get_true_byDeltaSogCog(self,preds, trues):
        preds[:, 0:1, :] = self.Correction_course(trues[:, 0:1, 2:] + preds[:, 0:1, :])
        posi_preds = self.sog_cog_position(trues[:, 0:1, :1], trues[:, 0:1, 1:2],
                                      preds[:, 0:1, :1], preds[:, 0:1, 1:])
        for i in range(1, len(preds[0])):
            preds[:, i:i + 1, :] = self.Correction_course(preds[:, i - 1:i, :] + preds[:, i:i + 1, :])
            posi_preds_ = self.sog_cog_position(posi_preds[:, i - 1:i, :1], posi_preds[:, i - 1:i, 1:2],
                                           preds[:, i:i + 1, :1], preds[:, i:i + 1, 1:])
            posi_preds = torch.cat([posi_preds, posi_preds_], 1)
        return posi_preds

    def forward(self, pred, true):
        preds= self.get_true_byDeltaSogCog(pred, true)
        # trues= self.get_true_byDeltaSogCog(truth,batch_true)
        # batch_true=batch_true[:,-24:,:2]
        return torch.mean(self.distance(preds[:, :, 0], preds[:, :, 1], true[:, 1:, 0], true[:, 1:, 1])), preds #计算后的真实位置
