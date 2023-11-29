import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Splat(nn.Module):
    def __init__(self, channels, radix, cardinality, reduction_factor=4):
        super(Splat, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels
        inter_channels = max(channels*radix//reduction_factor, 32)
        self.fc1 = nn.Conv2d(channels//radix, inter_channels, 1, groups=cardinality)
#         self.bn1 = nn.BatchNorm2d(inter_channels)
        self.bn1 = nn.InstanceNorm2d(inter_channels)
        self.relu = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=cardinality)
        self.rsoftmax = rSoftMax(radix, cardinality)

    def forward(self, x):
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 dilation=[1,3,5], groups=1, bias=True, 
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes%groups==0
        assert kernel_size==3, 'only support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        self.act = eval(act_layer)
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                bound = self.groups / (self.kernel_size**2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output
class AtrousSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, num_heads=2):
        super(AtrousSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.query_conv = ds_conv2d(in_channels, out_channels, kernel_size)
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels,1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, 1)
 
        self.dropout = nn.Dropout(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.query_conv.weight)
        nn.init.kaiming_uniform_(self.key_conv.weight)
        nn.init.kaiming_uniform_(self.value_conv.weight)
        nn.init.zeros_(self.query_conv.bias)
        nn.init.zeros_(self.key_conv.bias)
        nn.init.zeros_(self.value_conv.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        queries = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height*width).permute(0, 1, 3, 2)
        keys = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, height*width).permute(0, 1, 2, 3)
        values = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, height*width).permute(0, 1, 3, 2)
        energy = torch.matmul(queries, keys)
        attention = torch.softmax(energy / self.head_dim**0.5, dim=-1)
#         attention = self.dropout(attention)
        out = torch.matmul(attention, values)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, height, width)
        return out
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.2, with_depconv=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_depconv = with_depconv
        
        if self.with_depconv:
            self.fc1 = nn.Conv2d(
                in_features, hidden_features, 1, stride=1, padding=0, dilation=1, 
                groups=1, bias=True,
            )
            self.depconv = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, dilation=1, 
                groups=hidden_features, bias=True,
            )
            self.act = act_layer()
            self.fc2 = nn.Conv2d(
                hidden_features, out_features, 1, stride=1, padding=0, dilation=1, 
                groups=1, bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        if self.with_depconv:
            x = self.fc1(x)
            x = self.depconv(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
class CSA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, padding=1, stride=2,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5
        
        self.attn = nn.Linear(in_dim, kernel_size**4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        
        self.csa_group = 1
        assert out_dim % self.csa_group == 0
        self.weight = nn.Conv2d(
            self.kernel_size*self.kernel_size*out_dim, 
            self.kernel_size*self.kernel_size*out_dim, 
            1, 
            stride=1, padding=0, dilation=1, 
            groups=self.kernel_size*self.kernel_size*self.csa_group, 
            bias=qkv_bias,
        )
        assert qkv_bias == False
        fan_out = self.kernel_size*self.kernel_size*self.out_dim
        fan_out //= self.csa_group
        self.weight.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) # init
        
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, v=None):
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4) # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        v = x.permute(0, 3, 1, 2) # B,C,H, W
        v = self.unfold(v).reshape(
            B, self.out_dim, self.kernel_size*self.kernel_size, h*w
        ).permute(0,3,2,1).reshape(B*h*w, self.kernel_size*self.kernel_size*self.out_dim, 1, 1)
        v = self.weight(v)
        v = v.reshape(B, h*w, self.kernel_size*self.kernel_size, self.num_heads, 
                      self.out_dim//self.num_heads).permute(0,3,1,2,4).contiguous() # B,H,N,kxk,C/H
        
        x = (attn @ v).permute(0, 1, 4, 3, 2)
        x = x.reshape(B, self.out_dim * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, 
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)

class cross_attention(nn.Module):
    def __init__(self,in_channels) :
        super(cross_attention, self).__init__()
        self.queryConv =nn.Sequential(nn.Conv2d(in_channels, in_channels,1,1), 
                                      nn.GroupNorm(in_channels//4,in_channels)) 
        self.keyConv = nn.Sequential(nn.Conv2d(in_channels,in_channels,1,1), 
                                     nn.GroupNorm(in_channels//4,in_channels)) 
        self.psiConv = nn.Sequential(
            nn.GroupNorm(in_channels//4,in_channels),
            nn.Conv2d(in_channels,1,1,1), 
            nn.SiLU()) 
    def forward(self, query, key):
        value = key
        key = self.keyConv(key)
        query = self.queryConv(query)
        psi = F.silu(key+query)
        psi = self.psiConv(psi)
        return value*psi

class SPADE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SPADE, self).__init__()
        self.norm = nn.GroupNorm(in_channels//4,in_channels)
        self.share = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv_gamma = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv_beta = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x, segmap):
        normalized = self.norm(x)
        seg_share  = self.share(F.silu(self.norm(segmap)))
        gamma = self.conv_gamma(F.silu(self.norm(seg_share)))
        #beta = F.silu(self.conv_beta(self.norm(x)))
        return (1 + gamma) * normalized #+ beta


