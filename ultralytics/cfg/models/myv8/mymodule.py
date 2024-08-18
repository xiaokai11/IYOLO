import torch.nn.functional as F
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class Partial_conv3(nn.Module):

    def __init__(self,c1, c2):
        super().__init__()
        # //dim, 4. 'split_cat',
        n_div = 4
        forward = 'split_cat'
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x) :
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x
#ce1
class EnhancedUpsampleBlock(nn.Module):
    def __init__(self, c1, c2):

        super(EnhancedUpsampleBlock, self).__init__()
        input_channels = c1
        output_channels = c2
        # mid = c1 // 4
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            Partial_conv3(64, 64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, padding=1),
            Partial_conv3(output_channels, output_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Skip connection
        self.skip_connection = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            # Partial_conv3(output_channels, output_channels),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)

        # Decoder with skip connection
        decoded = self.decoder(encoded)
        skip = self.skip_connection(x)

        output = decoded + skip  # Adding skip connection

        return output

class block(nn.Module):
    # Channel-wise self-attention
    #ChannelAttentionLayer
    def __init__(self, input_channels, output_channels, num_heads=4):
        super(block, self).__init__()

        self.ChannelAttentionLayer = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(),
            nn.Linear(input_channels // 2, input_channels),
            nn.Sigmoid()  # Sigmoid activation for attention scores
        )

        # Spatial self-attention (Multi-head self-attention)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=input_channels, num_heads=num_heads)
        # B, C, H, W = x.size()
        # Depth-wise Separable Convolution
        self.depthwise_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1,
                                        groups=input_channels)

        # Layer normalization
        self.norm = nn.BatchNorm2d(input_channels)

    def forward(self, x):
        # Channel-wise self-attention
        channel_attention_weights = self.ChannelAttentionLayer(x.mean(-1).mean(-1))#1,256
        channel_attention_weights = channel_attention_weights.unsqueeze(-1).unsqueeze(-1) #1.256.1.1
        x = x * channel_attention_weights#1,256,16,16
        # Residual connection
        residual = x

        # Spatial self-attention (Multi-head self-attention)
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(2, 0, 1)  # 256,1,256
        x, _ = self.spatial_attention(x, x, x)#256,1,256;
        x = x.permute(1, 2, 0).view(B, C, H, W)  # 1,256,16,16

        # Residual connection
        x += residual

        # Depth-wise Separable Convolution
        x = self.depthwise_conv(x)#1,256,16,16

        # Layer normalization
        x = self.norm(x)

        return x
#myconnet
class TransChannelSpatializer(nn.Module):
    def __init__(self, c1, c2):

        super().__init__()
        in_channels = c1
        out_channels = c2
        # mid_channel = in_channels * 4

        self.dwconv = nn.Sequential(
            nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                        padding=1, groups=in_channels, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                        stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.lightmlp = block(in_channels, in_channels)


    def forward(self, x):
    # in 16,64,64  out 32,32,32 # in 32,32,32  out 64,16,16 # in 64,16,16  out 128,8,8
#目前进出一样
        #in 256,16,16
        x = self.dwconv(x)  #这里dwconv设置计算图像大小不能整除，但是输入64，32，16正好得到32，16，8
        x1 = self.lightmlp(x)
        x1 = self.conv2(x1)
        return x1


//
class ShuffleConvLayer(nn.Module):
    def __init__(self,c1,c2,):
        super().__init__()
        # exp_ratio = 6
        mid_channel = c1 // 8

        self.dwconv = nn.Sequential(
            GSConv(c1, mid_channel),  #gsconv 0.543，expratio
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel))

        self.conv2 = nn.Sequential(
            GSConv(mid_channel, c2),
            nn.ReLU(),
            nn.BatchNorm2d(c2))
        self.myshortcut = nn.Sequential(
                nn.AvgPool2d(1),
                nn.Conv2d(
                    in_channels=c1,
                    out_channels=c1,
                    kernel_size=1,
                    stride=1,
                    padding=0,),
            nn.ReLU(),
            nn.BatchNorm2d(c1))
    def forward(self, x):  #in 16，64，64  ou 32,32,32
        identity = x
        x = self.dwconv(x)
        x = self.conv2(x)
        y = self.myshortcut(identity)
        x = x + y
        return x

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d=1, act=True)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, d=1, act=True)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)




