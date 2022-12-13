import torch
import torch.nn as nn
from einops import rearrange
import math

def get_offset_tensor_chanel(x, y, c):
    x+=1
    y+=1
    kernel = torch.zeros(size=[1, 3, 3, 3])
    kernel[0][c][x][y] = 1.0
    return kernel

# [0,1] [1,0] [0,-1] [-1,0]
def get_cross_offset(c):
    #out = get_offset_tensor_chanel(0, 0, c)
    out = get_offset_tensor_chanel(0, 1, c)
    out = torch.cat([out, get_offset_tensor_chanel(1, 0, c)], dim=0)
    out = torch.cat([out, get_offset_tensor_chanel(0, -1, c)], dim=0)
    out = torch.cat([out, get_offset_tensor_chanel(-1, 0, c)], dim=0)
    return out

# https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/master/src/ffx-fsr2-api/shaders/ffx_fsr2_rcas.h
# https://www.shadertoy.com/view/7tfSWH
# ported from shader to tensors format
# probably not optimal way to execute on pytorch but at least it works with gradients
# could be made much better with cuda kernel (avoid transfer local memory <-> global memory as in StyleGan3)
# to do make rcas with 9 pixel pattern
class RCAS(nn.Module):
    def __init__(self):
        super(RCAS, self).__init__()
        self.register_buffer("r_cross_conv", get_cross_offset(0))
        #print(self.r_cross_conv.size())
        self.register_buffer("g_cross_conv", get_cross_offset(1))
        self.register_buffer("b_cross_conv", get_cross_offset(2))
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        # convert to space [0.0, 1.0]
        x_converted = (x + 1.0) * 0.5
        # small eps to avoid division by zero
        # clip grad during training for sure
        eps = 0.00001
        x_clamp = x_converted

        x_pad = self.pad(x_clamp)
        r_cross = nn.functional.conv2d(input=x_pad, weight=self.r_cross_conv, bias=None, stride=1)
        g_cross = nn.functional.conv2d(input=x_pad, weight=self.g_cross_conv, bias=None, stride=1)
        b_cross = nn.functional.conv2d(input=x_pad, weight=self.b_cross_conv, bias=None, stride=1)
        #print(r_cross.size(), "r_cross")
        r_max = torch.sum(nn.functional.softmax(r_cross, dim=1) * r_cross, dim=1).unsqueeze(dim=1)
        g_max = torch.sum(nn.functional.softmax(g_cross, dim=1) * g_cross, dim=1).unsqueeze(dim=1)
        b_max = torch.sum(nn.functional.softmax(b_cross, dim=1) * b_cross, dim=1).unsqueeze(dim=1)
        #print(r_max.size(), "r_max")
        r_min = torch.sum(nn.functional.softmin(r_cross, dim=1) * r_cross, dim=1).unsqueeze(dim=1)
        g_min = torch.sum(nn.functional.softmin(g_cross, dim=1) * g_cross, dim=1).unsqueeze(dim=1)
        b_min = torch.sum(nn.functional.softmin(b_cross, dim=1) * b_cross, dim=1).unsqueeze(dim=1)
        #print(r_min.size(), "r_min")
        r_sum = torch.sum(r_cross, dim=1).unsqueeze(dim=1)
        g_sum = torch.sum(g_cross, dim=1).unsqueeze(dim=1)
        b_sum = torch.sum(b_cross, dim=1).unsqueeze(dim=1)
        #print(r_sum.size(), "r_sum")
        sum_tensor = torch.cat([r_sum, g_sum, b_sum], dim=1)
        #print(sum_tensor.size(), "sum_tensor")
        r_edges = torch.cat([r_min / (r_max + eps), (1.0 - r_max) / (1.0 - r_min + eps)], dim=1)
        g_edges = torch.cat([g_min / (g_max + eps), (1.0 - g_max) / (1.0 - g_min + eps)], dim=1)
        b_edges = torch.cat([b_min / (b_max + eps), (1.0 - b_max) / (1.0 - b_min + eps)], dim=1)

        r_edge = -0.25 * torch.sum(nn.functional.softmin(r_edges, dim=1) * r_edges, dim=1).unsqueeze(dim=1)
        g_edge = -0.25 * torch.sum(nn.functional.softmin(g_edges, dim=1) * g_edges, dim=1).unsqueeze(dim=1)
        b_edge = -0.25 * torch.sum(nn.functional.softmin(b_edges, dim=1) * b_edges, dim=1).unsqueeze(dim=1)

        edges = torch.cat([r_edge, g_edge, b_edge], dim=1)
        edge = torch.sum(nn.functional.softmin(edges, dim=1) * edges, dim=1).unsqueeze(dim=1)
        w = edge * 0.36787944117
        out = (x_converted + sum_tensor * w) / (w * 4.0 + 1.0)

        # convert back
        out = out * 2.0 - 1.0
        return out

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