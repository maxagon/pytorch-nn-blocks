import torch
import torch.nn as nn
from einops import rearrange
import math

def sinc(x):
    if x == 0.0:
        return 1.0
    x = math.pi * x
    return math.sin(x) / x

def lanczos(x):
    if -3.0 <= x and x < 3.0:
        return sinc(x) * sinc(x / 3.0)
    return 0.0

def create_lanczos_kernel5(sub_pixel_offset_x, sub_pixel_offset_y):
    offset = [-2.0, -1.0, 0.0, 1.0, 2.0]
    kernel = torch.zeros(size=[1, 1, 5, 5])
    sum = 0.0
    for x in range(5):
        for y in range(5):
            ox = offset[x] + sub_pixel_offset_x
            oy = offset[y] + sub_pixel_offset_y
            weight = lanczos(ox) * lanczos(oy)
            sum += weight
            kernel[0][0][x][y] = weight
    kernel = kernel
    return kernel

def create_lanczos_upscale_weights():
    o1 = 0.5
    o2 = -0.5
    w =               create_lanczos_kernel5(o1, o1)
    w = torch.cat([w, create_lanczos_kernel5(o1, o2)], dim=0)
    w = torch.cat([w, create_lanczos_kernel5(o2, o1)], dim=0)
    w = torch.cat([w, create_lanczos_kernel5(o2, o2)], dim=0)
    return w

# https://en.wikipedia.org/wiki/Lanczos_resampling
# we can interpret each input channel as separate "batch" element so we can run single filter for each channel
# at least don't have to deal with cuda kernel
class LanczosSampler5(nn.Module):
    def __init__(self, channels):
        super(LanczosSampler5, self).__init__()
        self.register_buffer("filters", create_lanczos_upscale_weights())
        self.pad = nn.ReplicationPad2d(padding=(2,2,2,2))
        self.channels = channels
    def forward(self, x):
        bs = x.shape[0]
        out = x
        out = self.pad(out)
        out = rearrange(out, 'b c x y -> (b c) x y')
        out = torch.unsqueeze(out, dim=1)
        out = nn.functional.conv2d(out, self.filters)
        out = rearrange(out, '(b c) (nx ny) x y -> b c (x nx) (y ny)', nx=2, ny=2, b=bs)
        return out

class ConvBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_rad, pad_type='none', norm='none', nonlinearity='none', bias=True, dropout=0.0, stride=1, img_size=None, groups=1, group_norm_ch=32):
        super(ConvBlock2d, self).__init__()
        model = []

        # dropout
        if dropout != 0.0:
            model += [nn.Dropout2d(dropout)]

        # pad
        if (norm != 'none'):
            assert(not bias)
        if pad_type == 'replicate':
            model += [nn.ReplicationPad2d(kernel_rad)]
        elif pad_type == 'reflect':
            model += [nn.ReflectionPad2d(kernel_rad)]
        elif pad_type == 'zero':
            model += [nn.ZeroPad2d(kernel_rad)]
        elif pad_type != 'none':
            assert 0, "Wrong padding type: {}".format(pad_type)

        # conv
        kernel_size = kernel_rad * 2 + 1
        model += [nn.Conv2d(in_dim, out_dim, kernel_size, stride, bias=bias, groups=groups)]

        # normalization
        if norm == 'bn':
            model += [nn.BatchNorm2d(out_dim)]
        elif norm == 'in':
            model += [nn.InstanceNorm2d(out_dim)]
        elif norm == 'gn':
            model += [nn.GroupNorm(group_norm_ch, out_dim)]
        elif norm == 'ln':
            assert img_size != None, "Img size must be know for layer norm normalization"
            model += [nn.LayerNorm((out_dim, img_size, img_size))]
        elif norm != 'none':
            assert 0, "Wrong normalization: {}".format(norm)

        # activation
        if nonlinearity == 'relu':
            model += [nn.ReLU()]
        elif nonlinearity == 'lrelu':
            model += [nn.LeakyReLU(0.2)]
        elif nonlinearity == 'prelu':
            model += [nn.PReLU()]
        elif nonlinearity == 'selu':
            model += [nn.SELU()]
        elif nonlinearity == 'silu':
            model += [nn.SiLU()]
        elif nonlinearity == 'tanh':
            model += [nn.Tanh()]
        elif nonlinearity == 'gelu':
            model += [nn.GELU()]
        elif nonlinearity != 'none':
            assert 0, "Unsupported activation: {}".format(nonlinearity)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class DepthwiseConvBlock2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_rad, pad_type='none', norm='none', nonlinearity='none', bias=True, dropout=0.0, img_size=None, group_norm_ch=None):
        super(DepthwiseConvBlock2d, self).__init__()
        self.conv1 = ConvBlock2d(in_dim=in_dim, out_dim=in_dim, kernel_rad=kernel_rad,
            pad_type=pad_type, norm=norm, nonlinearity='none', bias=bias, dropout=0, stride=1, img_size=img_size, group_norm_ch=group_norm_ch, groups=in_dim)
        self.conv2 = ConvBlock2d(in_dim=in_dim, out_dim=out_dim, kernel_rad=0, pad_type='none',
            norm=norm, nonlinearity=nonlinearity, bias=bias, dropout=dropout, stride=1, groups=1, group_norm_ch=group_norm_ch)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class Upscale(nn.Module):
    def __init__(self):
        super(Upscale, self).__init__()
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

class Downscale(nn.Module):
    def __init__(self):
        super(Downscale, self).__init__()
    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

class ConvolutionDownscale(nn.Module):
    def __init__(self, in_dim, out_dim, group_norm_ch=32):
        super(ConvolutionDownscale, self).__init__()
        model = []
        model.append(nn.Conv2d(in_dim, out_dim, 2, 2, 0, 1, 1, bias=False))
        model.append(nn.GroupNorm(group_norm_ch, out_dim))
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class ConvolutionUpscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvolutionUpscale, self).__init__()
        model = []
        model.append(Upscale())
        model.append(ConvBlock2d(in_dim, out_dim, 1, 'zero', 'gn', bias=False))
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class AffineEmbeding(nn.Module):
    def __init__(self, emb_dim, in_dim):
        super(AffineEmbeding, self).__init__()
        self.emb = nn.Sequential(
            nn.Linear(emb_dim, in_dim*2)
        )
    def forward(self, x, emb):
        batch = x.shape[0]
        gamma, beta = self.emb(emb).view(batch, -1, 1, 1).chunk(2, dim=1)
        out = (1 + gamma) * x + beta
        return out