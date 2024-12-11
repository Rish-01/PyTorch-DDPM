import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(H * 2, W * 2), mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x
    
class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # Tensorflow 'same' is equivalent to padding 0 for these parameters and avgpool2d
    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.avgpool(x)
        return x
            
class ResBlock(nn.Module):
    def __init__(self, pos_embedding_dim, input_channels, output_channels, dropout_ratio, conv_shortcut=False):
        super().__init__()
        self.output_channels = output_channels
        self.conv_shortcut = conv_shortcut
        self.silu = nn.SiLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=pos_embedding_dim, out_features=output_channels)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(in_features=input_channels, out_features=output_channels)
        
    def forward(self, x, pos_emb):
        B, C, H, W = x.shape
        h = x
        
        h = self.silu(F.group_norm(h, num_groups=8))
        h = self.conv1(h)
        
        # Add positional embedding
        h += self.fc1(self.silu(pos_emb)).unsqueeze(2).unsqueeze(3)
        
        h = self.silu(F.group_norm(h, num_groups=8))
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.output_channels != C:
            if self.conv_shortcut:
                x = self.conv3(x)
            else:
                # BCHW -> BHWC for Linear on channel dim
                x = x.permute(0,2,3,1)
                x = self.fc2(x) 
                #Convert back BHWC -> BCHW 
                x = x.permute(0,3,1,2)
        
        return x + h
    
class AttentionBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.fc1 = nn.Linear(in_features=C, out_features=C)
        self.fc2 = nn.Linear(in_features=C, out_features=C)
        self.fc3 = nn.Linear(in_features=C, out_features=C)
        self.proj_out = nn.Linear(in_features=C, out_features=C)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = F.group_norm(x, num_groups=8)
        
        # BCHW -> BHWC for Linear on channel dim
        h = h.permute(0,2,3,1)
        
        # Extract query, key and value vectors
        q = self.fc1(h) 
        k = self.fc2(h)
        v = self.fc3(h)
        
        #Convert back BHWC -> BCHW 
        q = q.permute(0,3,1,2)
        k = k.permute(0,3,1,2)
        v = v.permute(0,3,1,2)

        # You can see hw and HW as seq len in the original transformers. C is the embedding dim.
        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (int(self.C) ** (-0.5))
        w = w.view(B, H, W, H * W)
        w = F.softmax(w, dim=-1)
        w = w.view(B, H, W, H, W)
        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        
        # BCHW -> BHWC for Linear on channel dim
        h = h.permute(0,2,3,1)
        
        h = self.proj_out(h)
        
        #Convert back BHWC -> BCHW 
        h = h.permute(0,3,1,2)
        
        assert h.shape == x.shape
        return x + h


class UNet(nn.Module):
    """"
    timesteps: number of steps in the markov chain.
    embedding_dim: 'ch' from the original paper.
    ch_mult: multiple
    """
    def __init__(self, timesteps, embedding_dim, out_channel=3, num_res_blocks=2, ch_mult=(1, 2, 4, 8), dropout=0.):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_dim = embedding_dim
        self.num_res_blocks = num_res_blocks
        self.ch_mult = ch_mult
        self.dropout = dropout
        self.out_channel = out_channel
        self.num_resolutions = len(ch_mult)
        self.position_embedding = self.calcPositionalEmbedding(timesteps, self.embedding_dim)
    
        
        self.time_mlp = nn.Sequential(
                            nn.Linear(in_features=embedding_dim, out_features=embedding_dim*4),
                            nn.SiLU(),
                            nn.Linear(in_features=embedding_dim*4, out_features=embedding_dim*4)    
                        )
        
        dims = [embedding_dim, *map(lambda m: embedding_dim * m, ch_mult)] 
        self.in_out = list(zip(dims[:-1], dims[1:])) # map in channels to out channels
        mid_dim = dims[-1]
        
        self.conv = nn.Conv2d(in_channels=self.embedding_dim * self.ch_mult[0], out_channels=out_channel, kernel_size=3, stride=1, padding=0)
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=3, stride=1, padding=1)
        
        self.final_resblock = ResBlock(pos_embedding_dim=embedding_dim * 4, 
                                input_channels=embedding_dim * 2,
                                output_channels=embedding_dim,
                                dropout_ratio=self.dropout, 
                                conv_shortcut=False)
        self.final_conv = nn.Conv2d(in_channels=embedding_dim, out_channels=3, kernel_size=3, padding=1)
        
        self.downsample_block = self.downsampleBlock()
        self.middle_block = self.middleBlock(mid_dim)
        self.upsample_block = self.upsampleBlock()
        
    def downsampleBlock(self):
        downsample_blocks = nn.ModuleList()
        
        for (dim_in, dim_out) in self.in_out:
            downsample_blocks.append(nn.ModuleList([
                ResBlock(self.embedding_dim * 4, 
                            input_channels=dim_in,
                            output_channels=dim_in,
                            dropout_ratio=self.dropout, 
                            conv_shortcut=False),
                
                ResBlock(pos_embedding_dim=self.embedding_dim * 4, 
                            input_channels=dim_in,
                            output_channels=dim_in,
                            dropout_ratio=self.dropout, 
                            conv_shortcut=False),
                
                AttentionBlock(dim_in),
                DownSample(in_channel=dim_in, out_channel=dim_out, with_conv=True)
            ]))
                
        return downsample_blocks
    
    def middleBlock(self, mid_dim):
        resblock1 = ResBlock(pos_embedding_dim=self.embedding_dim * 4, 
                                input_channels=mid_dim,
                                output_channels=mid_dim,
                                dropout_ratio=self.dropout, 
                                conv_shortcut=False)
        
        attention1 = AttentionBlock(mid_dim)
        
        resblock2 = ResBlock(pos_embedding_dim=self.embedding_dim * 4, 
                                input_channels=mid_dim,
                                output_channels=mid_dim,
                                dropout_ratio=self.dropout, 
                                conv_shortcut=False)
        upsample = Upsample(in_channel=mid_dim, out_channel=mid_dim, with_conv=True)
        # conv_layer=nn.Conv2d(in_channels= mid_dim,out_channels=mid_dim//2,kernel_size=3,padding=1)
        
        return nn.ModuleList([resblock1, attention1, resblock2, upsample])
        
    
    def upsampleBlock(self):
        upsample_blocks = nn.ModuleList()
        # in_channel = self.upsample_input_channel    
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            upsample_blocks.append(nn.ModuleList([
                ResBlock(self.embedding_dim * 4, 
                            input_channels=dim_in+dim_out,
                            output_channels=dim_out,
                            dropout_ratio=self.dropout, 
                            conv_shortcut=False),
                
                ResBlock(pos_embedding_dim=self.embedding_dim * 4, 
                            input_channels=dim_in+dim_out,
                            output_channels=dim_out,
                            dropout_ratio=self.dropout, 
                            conv_shortcut=False),
                
                AttentionBlock(dim_out),
                Upsample(in_channel=dim_out, out_channel=dim_in, with_conv=True)
            ]))
                
        return upsample_blocks
                
    def calcPositionalEmbedding(self, timesteps, embedding_dim):
        pos = torch.arange(0, timesteps).unsqueeze(1)
        div_term = torch.pow(10000., -torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim)
        
        # We take values from the same domain point from each sinusoid for a particular word 
        # The number of sinusoids we sample from is equal to the embedding size
        # We alternate between sin and cosine sinusoids to represent the embedding
        positional_encoding = torch.zeros((timesteps, embedding_dim))
        positional_encoding[:, 0::2] = torch.sin(pos * div_term) # sin at even positions
        positional_encoding[:, 1::2] = torch.cos(pos * div_term) # cos at odd positions
        return positional_encoding.to(self.device)
        
    def forward(self, x):
        B, C, H, W = x.shape
        pos_emb = self.time_mlp(self.position_embedding)
        assert list(pos_emb.shape) == [B, self.embedding_dim * 4] 
        hs = []
        x = self.init_conv(x)
        x_initial = x.clone()
        
        # Downsample blocks
        for i, (resblock1, resblock2, attention, downsample) in enumerate(self.downsample_block):
            x = resblock1(x, pos_emb)
            hs.append(x)
            x = resblock2(x, pos_emb)
            x = attention(x) + x
            hs.append(x)
            x = downsample(x)
            
        # Middle Blocks
        for i, block in enumerate(self.middle_block):
            # Odd Even position blocks are resblocks
            if i % 2 == 0:
                x = block(x, pos_emb)
            else:
                x = block(x)
        
        # Upsample Blocks 
        for i, (resblock1, resblock2, attention, upsample) in enumerate(self.upsample_block):
            x = resblock1(torch.cat([x, hs.pop()], dim=1), pos_emb)
            x = resblock2(torch.cat([x, hs.pop()], dim=1), pos_emb)
            x = attention(x)
            if i != len(self.upsample_block)-1:
                x = upsample(x)
            
        x = self.final_resblock(torch.cat([x_initial, x], dim=1), pos_emb)
        x = self.final_conv(x)
        return x
    

if __name__=="__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(5, 3, 128,128).to(device)

    unet = UNet(timesteps=5, embedding_dim=128, out_channel=3, num_res_blocks=2, ch_mult=(1, 2, 4, 8), dropout=0).to(device)
    x = unet(x)
