import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import sys
sys.path.append("./models")
from basic import *

class style_encoder(nn.Module):
    def __init__(self,in_channels,embed_dim ):
        super(style_encoder,self).__init__()
        self.embed_dim =embed_dim 
        self.styleconv1 = nn.Sequential(
            nn.Conv2d(1, self.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(self.embed_dim//4,self.embed_dim),
            nn.SiLU(inplace=True)
        )
        self.styleconv2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(self.embed_dim//4,self.embed_dim),
            nn.SiLU(inplace=True)
        )
            
        self.styleconv3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim*2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(self.embed_dim//2,self.embed_dim*2),
            nn.SiLU(inplace=True)
        )
        self.styleconv4 = nn.Sequential(
            nn.Conv2d(self.embed_dim*2, self.embed_dim*4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(self.embed_dim//2,self.embed_dim*4),
            nn.SiLU(inplace=True)
        )
        self.upUpsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.styleup4 = nn.Sequential(
            nn.Conv2d(self.embed_dim*6,self.embed_dim*2, 3,1,1),
            nn.GroupNorm(self.embed_dim//2,self.embed_dim*2),
            nn.SiLU(inplace=True)
        )
        self.styleup3 = nn.Sequential(
            nn.Conv2d(self.embed_dim*3,self.embed_dim*1, 3,1,1),
            nn.GroupNorm(self.embed_dim//4,self.embed_dim*1),
            nn.SiLU(inplace=True)
        )
        self.styleup2 = nn.Sequential(
            nn.Conv2d(self.embed_dim*2,self.embed_dim*1,3,1,1),
            nn.GroupNorm(self.embed_dim//4,self.embed_dim*1),
            nn.SiLU(inplace=True)
        )
        
    def forward (self,x):
        x1 = self.styleconv1(x)
        x2 = self.styleconv2(x1)
        x3 = self.styleconv3(x2)
        x4 = self.styleconv4(x3)
        
        x5 = self.upUpsample(x4)
        x5 = torch.cat((x5,x3),dim=1)
        x5 = self.styleup4(x5)
        
        x6 = self.upUpsample(x5)
        x6 = torch.cat((x6,x2),dim=1)
        x6 = self.styleup3(x6)
        
        x7 = self.upUpsample(x6)
        x7 = torch.cat((x7,x1),dim=1)
        x7 = self.styleup2(x7)
        
        return x7, x6, x5, x4

    
class TGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.overlappatch_size=48
        self.patch_size = 96
        self.num_patch=(384//self.patch_size)
        self.num_patches=self.num_patch**2
        self.embed_dim = 64
        
        
        self.patches = nn.Sequential(
            nn.Conv2d(1, self.embed_dim,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(self.embed_dim//4,self.embed_dim),
            nn.SiLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(self.embed_dim//4,self.embed_dim),
            nn.SiLU(inplace=True),
        )
        
        
        # self.cross_attention1 = cross_attention(in_channels=self.embed_dim, num_heads=2)
        self.cross_attention1 = cross_attention(in_channels=self.embed_dim)
        self.attn1 = CSA(in_dim=self.embed_dim, out_dim=self.embed_dim, num_heads=2)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.embed_dim, self.patch_size,self.patch_size))
        self.norm1 = nn.GroupNorm(self.embed_dim//4,self.embed_dim)
        self.mlp1 = Mlp(self.embed_dim,self.embed_dim*4)
        # self.ResidualBlock1 = ResidualBlock(self.embed_dim,self.embed_dim)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim*2,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32,self.embed_dim*2),
            nn.SiLU(inplace=True),
        )
        
        # self.cross_attention2 = cross_attention(in_channels=self.embed_dim*2, num_heads=2)
        self.cross_attention2 = cross_attention(in_channels=self.embed_dim*2)
        self.attn2 = AtrousSelfAttention(self.embed_dim*2, self.embed_dim*2, kernel_size=3, dilation=1, num_heads=4)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.embed_dim*2, self.patch_size//2, self.patch_size//2))
        self.norm2 =nn.GroupNorm(32,self.embed_dim*2)
        self.mlp2 = Mlp(self.embed_dim*2,self.embed_dim*2*8)
        # self.ResidualBlock2 = ResidualBlock(self.embed_dim*2,self.embed_dim*2)

        
        self.down3 = nn.Sequential(
            nn.Conv2d(self.embed_dim*2, self.embed_dim*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32,self.embed_dim*4),
            nn.SiLU(inplace=True),
        )
        
        # self.cross_attention3 = cross_attention(in_channels=self.embed_dim*4, num_heads=2)
        self.cross_attention3 = cross_attention(in_channels=self.embed_dim*4)
        self.attn3 = AtrousSelfAttention(self.embed_dim*4, self.embed_dim*4, kernel_size=3, dilation=1, num_heads=4)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.embed_dim*4, self.patch_size//4, self.patch_size//4))
        self.norm3 = nn.GroupNorm(32,self.embed_dim*4)
        self.mlp3 = Mlp(self.embed_dim*4,self.embed_dim*4*4)

        
        self.down4 = nn.Sequential(
            nn.Conv2d(self.embed_dim*4, self.embed_dim*8,  kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32,self.embed_dim*8),
            nn.SiLU(inplace=True),
        )
        
        # self.cross_attention4 = cross_attention(in_channels=self.embed_dim*1, num_heads=2)
        self.cross_attention4 = cross_attention(in_channels=self.embed_dim*1)
        self.attn4 = AtrousSelfAttention(self.embed_dim*8, self.embed_dim*8, kernel_size=3, dilation=1, num_heads=4)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.embed_dim*8, self.patch_size//8, self.patch_size//8))
        self.norm4 = nn.GroupNorm(self.embed_dim//2,self.embed_dim*8)
        self.mlp4 = Mlp(self.embed_dim*8,self.embed_dim*8*4)


        self.upUpsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # self.threethree = SPADE(self.embed_dim*4, self.embed_dim*4)
        self.con3 = nn.Sequential(
            nn.GroupNorm(32,self.embed_dim*8),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*8,self.embed_dim*4, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(0.2)
        )
        self.con33 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GroupNorm(32,self.embed_dim*8),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*8,self.embed_dim*4, kernel_size=1, bias=False),
            # nn.Conv2d(self.embed_dim*8,self.embed_dim*4, kernel_size=1),
             nn.Dropout(0.2)
        )
            
        self.con2 = nn.Sequential(
            nn.GroupNorm(32,self.embed_dim*4),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*4,self.embed_dim*2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(0.2)      
        )
        self.con22 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GroupNorm(32,self.embed_dim*12),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*12, self.embed_dim*2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(0.2)
            
        )


        
        #48->96

        self.oneone =SPADE(self.embed_dim*1, self.embed_dim*1)
        self.con11 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GroupNorm(32,self.embed_dim*10),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*10, self.embed_dim*1,kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(0.2)
        )
        self.con1 = nn.Sequential(
            nn.GroupNorm(32,self.embed_dim*2),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*2,self.embed_dim*1, kernel_size=1),
            nn.Dropout(0.2)
        )
        # self.con11 = nn.Conv2d(self.embed_dim*2,self.embed_dim*1, kernel_size=3, stride=1, padding=1)
        

        self.upup =  SPADE(self.embed_dim*1, self.embed_dim*1)
        self.conupup =  nn.Conv2d(self.embed_dim*1,self.embed_dim*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conupupnorm = nn.GroupNorm(self.embed_dim//4,self.embed_dim)
        
        self.transup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GroupNorm(32,self.embed_dim*5),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*5, self.embed_dim*5,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.2)
        )
        
        self.conup = nn.Sequential(
            nn.GroupNorm(32,self.embed_dim*6),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*6,self.embed_dim*1, kernel_size=1, bias=True),
        )
        self.transout = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GroupNorm(32,self.embed_dim*2),
            nn.SiLU(True),
            nn.Conv2d(self.embed_dim*2, self.embed_dim*1,kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.2)
        )
        
        self.output = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(self.embed_dim*1, 1, 3,1,1, bias=False),
            # nn.Tanh(),
        )   
        
        self.gelu=nn.GELU()
        drop_path=0.2
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.transformer_layer = 2
        self.style = style_encoder(1,self.embed_dim)
        
    def forward(self,x,style):
        
        style1, style2, style3, style4 = self.style(style)
        
        # x_deblur = self.deblur(x)
        #input 1*384*384-> 16*96*96
        x1 = self.patches(x)
        x2t = self.down1(x1)
        #x2r = self.ResidualBlock1(x2t)
        x2 = x2t + self.pos_embed1
        for i in range(self.transformer_layer):
            x2 = x2 + self.drop_path(self.attn1(self.norm1(x2)))
            x2 = F.silu(x2)
        x2 = x2 + self.drop_path(self.mlp1(self.norm1(x2)))
        
        #16*96*96 -> 64*48*48
        x3t = self.down2(x2)
        #x3r = self.ResidualBlock2(x3t)
        x3 = x3t + self.pos_embed2
        for i in range(self.transformer_layer):
            x3 = x3 + self.drop_path(self.attn2(self.norm2(x3)))
            x3 = F.silu(x3)
        x3 = x3 + self.drop_path(self.mlp2(self.norm2(x3)))

        
        #64*48*48 ->256*24*24
        x4t = self.down3(x3)
        #x4r = self.ResidualBlock3(x4t)
        x4 = x4t + self.pos_embed3
        for i in range(self.transformer_layer):
            x4 = x4 + self.drop_path(self.attn3(self.norm3(x4)))
            x4 = F.silu(x4)
        x4 = x4 + self.drop_path(self.mlp3(self.norm3(x4)))

        x5t = self.down4(x4)
        #x5r = self.ResidualBlock4(x5t)
        x5 = x5t + self.pos_embed4
        for i in range(self.transformer_layer):
            x5 = x5 + self.drop_path(self.attn4(self.norm4(x5)))
            x5 = F.silu(x5)
        x5 = x5 + self.drop_path(self.mlp4(self.norm4(x5)))
        
        # x6 = self.upUpsample(x5)
        x6 = self.con33(x5)
        # x4_cross = self.cross_attention3(x6,x4,x4)
        x4_cross = self.cross_attention3(x6,x4)
        x6_cat = torch.cat((x6,x4_cross),dim=1)
        x6_cat1 = self.con3(x6_cat)
        x6_cat_cross = self.cross_attention3(x6,style4)
        x6_cat = torch.cat((x6_cat1,x6_cat1,x6_cat_cross),dim=1)

        x7 = self.con22(x6_cat)
        # x3_cross = self.cross_attention2(x7,x3,x3)
        x3_cross = self.cross_attention2(x7,x3)
        x7_cat = torch.cat((x7,x3_cross),dim=1)
        x7_cat1 = self.con2(x7_cat)
        # x7_cat_cross = self.cross_attention2(x7_cat1,style3,style3)
        x7_cat_cross = self.cross_attention2(x7,style3)
        x7_cat = torch.cat((x7_cat1,x7_cat1,x7_cat1,x7_cat_cross,x7_cat_cross),dim=1)

        x8 = self.con11(x7_cat)
        # x2_cross = self.cross_attention1(x8,x2,x2)
        x2_cross = self.cross_attention1(x8,x2)
        x8_cat = torch.cat((x8,x2_cross),dim=1)
        x8_cat1 = self.con1(x8_cat)
        # x8_cat_cross = self.cross_attention1(x8_cat1,style2,style2)
        x8_cat_cross = self.cross_attention1(x8,style2)
        x8_cat = torch.cat((x8_cat1,x8_cat1,x8_cat_cross,x8_cat_cross,x8_cat_cross),dim=1)
        # x8_cat = x8_cat1 + F.silu(self.norm1(self.con11(x8_cat)))
        
        x9 = self.transup(x8_cat)
        x9_cat = torch.cat((x9,x1),dim=1)
        x9_cat = self.conup(x9_cat)
        x9_cat_cross = self.cross_attention4(x9_cat,style1)
        x9_cat1 = torch.cat((x9_cat,x9_cat_cross),dim=1)
        
        x11 = self.transout(x9_cat1)
        x11 = self.output(x11)
        return x11


