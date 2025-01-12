import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual

class iAFFM(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFFM, self).__init__()

        self.weight = CSAM1d(channels=channels, r=r, enable_sa=True)
        self.weight2 = CSAM1d(channels=channels, r=r, enable_sa=True)

    def forward(self, x, residual):
        xa = x + residual
        wei = self.weight(xa)
        xi = x * wei + residual * (1 - wei)

        wei2 = self.weight2(xi)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class AFFM(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFFM, self).__init__()
        self.weight = CSAM1d(channels=channels, r=r, enable_sa=True)
        self.channel_mappper = nn.Conv1d(channels, channels//2, kernel_size=1, stride=1, padding=0)

    def forward(self, x, residual):
        xa = torch.cat([x, residual], dim=1)
        wei = self.weight(xa)
        wei = self.channel_mappper(wei)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class CSAM1d(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4, enable_sa=True):
        super(CSAM1d, self).__init__()
        inter_channels = int(channels // r)
        self.enable_sa = enable_sa

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # self.sigmoid = nn.Sigmoid()
        for m in self.local_att.modules():
            if isinstance(m, nn.Conv1d):
                normal_init(m, std=0.001)
        for m in self.global_att.modules():
            if isinstance(m, nn.Conv1d):
                normal_init(m, std=0.001)

        if self.enable_sa:
            self.spatial_attention = nn.Conv1d(
                2,
                1,
                7,
                stride=1,
                padding=3,
                groups=1,
                bias=True)
            self.gamma = nn.Parameter(
                1e-6 * torch.ones((1, channels, 1)), requires_grad=True)
            
            for m in self.spatial_attention.modules():
                if isinstance(m, nn.Conv1d):
                    normal_init(m, std=0.001)
            

    def forward(self, x):
        wei_xl = self.local_att(x)
        wei_xg = self.global_att(x)
        wei = wei_xl + wei_xg
        # wei = self.sigmoid(xlg)
        feat = x * wei
        if self.enable_sa:
            avg_feat = torch.mean(x, dim=1, keepdim=True)
            max_feat, _ = torch.max(x, dim=1, keepdim=True)
            spatial_weight = torch.cat([avg_feat, max_feat], dim=1)
            spatial_weight = self.spatial_attention(spatial_weight)
            feat = feat + self.gamma * x * spatial_weight
            
        return feat


class CSAM2d(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, out_channels=256, r=4, enable_sa=True):
        super(CSAM2d, self).__init__()

        # self.channel_aggregator = ConvModule(
        #     in_channels,
        #     out_channels,
        #     1,
        #     conv_cfg=None,
        #     norm_cfg=dict(type='GN', num_groups=32),
        #     act_cfg=None)

        inter_channels = int(out_channels // r)
        self.enable_sa = enable_sa

        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        # self.sigmoid = nn.Sigmoid()

        if self.enable_sa:
            self.spatial_attention = nn.Conv2d(
                2,
                1,
                7,
                stride=1,
                padding=3,
                groups=1,
                bias=True)
            self.gamma = nn.Parameter(
                1e-6 * torch.ones((1, out_channels, 1, 1)), requires_grad=True)
        

    def forward(self, x):

        # x = self.channel_aggregator(x)

        wei_xl = self.local_att(x)
        wei_xg = self.global_att(x)
        wei = wei_xl + wei_xg
        # wei = self.sigmoid(xlg)
        feat = x * wei
        if self.enable_sa:
            avg_feat = torch.mean(x, dim=1, keepdim=True)
            max_feat, _ = torch.max(x, dim=1, keepdim=True)
            spatial_weight = torch.cat([avg_feat, max_feat], dim=1)
            spatial_weight = self.spatial_attention(spatial_weight)
            feat = feat + self.gamma * x * spatial_weight
            
        return feat
