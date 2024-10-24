import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channel=64, channel_mult=[1,2,4,8]): 
        super(UNet, self).__init__()
        # in_channels = 3 for RGB images
        self.in_channels = in_channels
        # out_channels = 1 for mask output
        self.out_channels = out_channels
        # base channel count for the model, channel=64 by default
        self.channel = channel
        # channel multiplier for each level of the UNet channel_mult=(1,2,4,8)
        self.channel_mult = channel_mult

        self.c1 = Conv_Block(in_channels, channel)
        self.d1 = DownSample(channel)

        self.c2 = Conv_Block(channel, channel*2)
        self.d2 = DownSample(channel*2)

        self.c3 = Conv_Block(channel*2, channel*4)
        self.d3 = DownSample(channel*4)


        self.c4 = Conv_Block(channel*4, channel*8)


        self.u1 = UpSample(channel*8)
        self.c5 = Conv_Block(channel*8, channel*4)
        
        self.u2 = UpSample(channel*4)
        self.c6 = Conv_Block(channel*4, channel*2)

        self.u3 = UpSample(channel*2)
        self.c7 = Conv_Block(channel*2, channel)

        self.out = nn.Sequential(
            nn.Conv2d(channel,out_channels,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        O1 = self.c5(self.u1(R4, R3))
        O2 = self.c6(self.u2(O1, R2))
        O3 = self.c7(self.u3(O2, R1))
        out = self.out(O3)

        return out