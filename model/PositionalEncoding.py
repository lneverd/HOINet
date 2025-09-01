import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, channels, window_size, token_length):
        super().__init__()

        pos_list = []
        for tk in range(window_size[0]):
            for st in range(window_size[1]):
                for pk in range(window_size[2]):
                    for tl in range(token_length):
                        pos_list.append(tl)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(window_size[0] * window_size[1] * window_size[2] * token_length, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels)) 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(window_size[0], window_size[1] * window_size[2], token_length, channels).permute(3, 0, 1, 2).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # n c t ej tokenlength
        x = self.pe[:, :, :, :x.size(3)]
        return x

class ResNetPositionalEncoding(nn.Module):
    def __init__(self, channels, time, height, width):
        super().__init__()
        # 展开后的token数量
        self.token_length = time * height * width
        pe = torch.zeros(self.token_length, channels)
        position = torch.arange(0, self.token_length, dtype=torch.float).unsqueeze(1)
        # 使用与Transformer中类似的编码方式
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float) * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状: (1, token_length, channels)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状: (batch, token_length, channels)
        # 注意：这里假设x的token数量与预设的token_length一致
        x = self.pe[:, :x.size(1), :]
        return x
class FusedPositionalEncoding(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, channels)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float) * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状: (1, seq_len, channels)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状: (batch, seq_len, channels)
        x = self.pe[:, :x.size(1), :]
        return x