import torch.nn as nn
from .TSABlock import TSABlock
import torch
import torch.nn.functional as F
from math import ceil
import numpy as np
from .resnet import resnet50
from .transformer_bimodal import Transformer_Encoder
from .PositionalEncoding import FusedPositionalEncoding
# Notation
# N: batch size
# C: coordinates (channel dimension)
# T: time (frame numbers)
# J: joint numbers (denoted as V in feeders)
# E: entity numbers (denoted as M in feeders)
# TokenNum(U): token numbers


def window_partition(x, window_size):
    """
    Args:
        x: (N, C, T, J, E)
        window_size (tuple[int]): window size (Tw, Jw, Ew)
    Returns:
        windows: (N, C, Tw, Jw * Ew, TokenNum)20 1 3
        pad (left, right, top, bottom, front, back)
    """
    N, C, T, J, E = x.shape
    pad_T1 = pad_J1 = pad_E1 = 0
    pad_T2 = (window_size[0] - T % window_size[0]) % window_size[0]
    pad_J2 = (window_size[1] - J % window_size[1]) % window_size[1]
    pad_E2 = (window_size[2] - E % window_size[2]) % window_size[2]
    x = F.pad(x, (pad_E1, pad_E2, pad_J1, pad_J2, pad_T1, pad_T2), mode='replicate')

    N, C, T, J, E = x.shape

    x = x.contiguous().view(N, C, window_size[0], T // window_size[0], window_size[1], J // window_size[1], window_size[2], E // window_size[2])
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous().view(N, C, window_size[0], -1, (T // window_size[0]) * (J // window_size[1]) * (E // window_size[2]))
    return x

# ISTA-Net
class Model(nn.Module):
    def __init__(self, window_size, num_classes, num_joints, 
                 num_frames, num_persons, num_heads,num_objs,num_verbs, num_channels,
                 kernel_size, use_pes=True, config=None, 
                 att_drop=0.2, dropout=0.2):
        super().__init__()

        self.in_channels = config[0][0]
        self.out_channels = config[-1][1]

        self.window_size = window_size
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_persons = num_persons
        self.num_tokens = ceil(num_frames / self.window_size[0]) * ceil(num_joints / self.window_size[1]) * ceil(num_persons / self.window_size[2])

        self.embed = nn.Sequential(
            nn.Conv3d(num_channels, self.in_channels, 1),
            nn.BatchNorm3d(self.in_channels),
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(TSABlock(in_channels, out_channels, qkv_dim, 
                                         window_size=self.window_size,
                                         num_tokens=self.num_tokens,
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))
        self.transformerEncoder = Transformer_Encoder(d_model=self.out_channels*2,nhead=4,num_encoder_layers=1,dim_feedforward=1024,dropout=att_drop)
        self.drop_out = nn.Dropout(dropout)
        self.rgb_proj = nn.Sequential(nn.Linear(2048,1024),
                                  nn.Linear(1024,512),
                                  nn.Linear(512,256))
        self.pose_proj = nn.Linear(self.out_channels,self.out_channels)
        self.conv = nn.Conv3d(self.out_channels, self.out_channels, (1, num_joints, num_persons), (1, num_joints, num_persons), 0)#如果说合起来的话还真不能用avgadaptivepool nn.AdaptiveAvgPool3d((120, 1, 1))
        self.resnet = resnet50(pretrained=False)
        self.res_pose_proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1)
        )
        self.res_rgb_proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1)
        )
        self.posergb_proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 2, 256 * 2),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1)
        )
        self.feat_anticipation_mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 2, 256 * 2),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 2, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024,2048)
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifer = nn.Linear(512, num_classes)
        self.fusedposencoding = FusedPositionalEncoding(512,num_frames)
        self.cls_obj = nn.Linear(512,num_objs)
        self.cls_verb = nn.Linear(512,num_verbs)

    def forward(self, x,img_list):

        N, C, T, J, E = x.shape
        batch_size, time, channel, width, height = img_list.shape
        #x (N, C, Tw, Jw * Ew, TokenNum) 1 3 20 3 126
        x = window_partition(x, window_size=self.window_size)
        x = self.embed(x)
        img_list = img_list.contiguous().view(-1, channel, width, height)
        features,feature_map = self.resnet(img_list) #整成batch tokennum channel形状
        BT,out_channel = features.shape
        features = features.view(batch_size, time, -1).contiguous()
        features = F.normalize(features,dim=-1)
        rgb_action_feat = self.rgb_proj(features.view(BT,-1)).view(batch_size,time,-1)

        rgb_action_feat = F.normalize(rgb_action_feat,dim=-1) #rgb action特征
        for i, block in enumerate(self.blocks):
            x = block(x)

        x = x.view(N, self.out_channels, self.window_size[0], self.window_size[1]*self.window_size[2],
                   ceil(self.num_frames / self.window_size[0]),
                   ceil(self.num_joints / self.window_size[1]),
                   ceil(self.num_persons / self.window_size[2])).permute(0, 1, 4, 2, 3, 5, 6).contiguous().view(N,self.out_channels,self.num_frames, self.num_joints, self.num_persons)
        x = self.conv(x)
        x = x.view(N, self.out_channels, -1)
        x = x.permute(0, 2, 1).contiguous()
        pose_action_feat = self.pose_proj(x)

        pose_action_feat = F.normalize(pose_action_feat,dim=-1)
        fused = torch.cat([pose_action_feat,rgb_action_feat],dim=-1)
        rgb_action_feat = self.res_rgb_proj(rgb_action_feat)
        pose_action_feat = self.res_pose_proj(pose_action_feat)
        posrgbfeature = self.posergb_proj(fused)

        rgb_feature_for_next = self.feat_anticipation_mlp(posrgbfeature)
        reconst_rgb_feat = rgb_feature_for_next[:, 0:T - 1, :]
        original_rgb_feat = features[:, 1:T, :]
        l1_loss = F.l1_loss(reconst_rgb_feat,original_rgb_feat)

        posrgbfeature = posrgbfeature+torch.cat([pose_action_feat,rgb_action_feat],dim=-1)
        # TODO 融合之后添加Posencoding
        posrgbfeature_pos = self.fusedposencoding(posrgbfeature)
        out = self.transformerEncoder(posrgbfeature,src_pos=posrgbfeature_pos)[0]
        out = out.permute(0,2,1).contiguous()
        out = self.pool(out)
        out = out.view(N,-1)
        obj_logits = self.cls_obj(out)
        verb_logits = self.cls_verb(out)
        out = self.classifer(out)
        return obj_logits,verb_logits,out,l1_loss.unsqueeze(0)
