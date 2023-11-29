import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("./models")
from basic import *
#FPSE discriminator 
class TDiscriminator(nn.Module):
    def __init__(self, input_nc=1, nf=64, label_nc=1):
        super(TDiscriminator, self).__init__()
        # bottom-up pathway
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(nf//4,nf),
            nn.SiLU(True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32,nf*2),
            nn.SiLU(True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32,nf*4),
            nn.SiLU(True))
        self.enc4 = nn.Sequential(
            nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32,nf*8),
            nn.SiLU(True))
        self.enc5 = nn.Sequential(
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32,nf*8),
            nn.SiLU(True))

        # top-down pathway
        self.lat2 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=1),
            nn.GroupNorm(32,nf*4),
            nn.SiLU(True))
        self.lat3 = nn.Sequential(
            nn.Conv2d(nf*4, nf*4, kernel_size=1),
            nn.GroupNorm(32,nf*4),
            nn.SiLU(True))
        self.lat4 = nn.Sequential(
            nn.Conv2d(nf*8, nf*4, kernel_size=1),
            nn.GroupNorm(32,nf*4),
            nn.SiLU(True))
        self.lat5 = nn.Sequential(
            nn.Conv2d(nf*8, nf*4, kernel_size=1),
            nn.GroupNorm(32,nf*4),
            nn.SiLU(True))
        
        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # final layers
        self.final2 = nn.Sequential(
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(nf//2,nf*2),
            nn.SiLU(True))
        self.final3 = nn.Sequential(
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(nf//2,nf*2),
            nn.SiLU(True))
        self.final4 = nn.Sequential(
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(nf//2,nf*2),
            nn.SiLU(True))
    
        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)
        
        self.L1 = nn.L1Loss()

    def forward(self, fake_and_real_img, segmap):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)  #384->192
        feat12 = self.enc2(feat11)   #192->96
        feat13 = self.enc3(feat12)   #96->48
        feat14 = self.enc4(feat13)  #48->24
        feat15 = self.enc5(feat14)   #24->12
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)  #24
        feat23 = self.up(feat24) + self.lat3(feat13)  #48
        feat22 = self.up(feat23) + self.lat2(feat12)  #96
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segemb = self.embedding(segmap)
        segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
        segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]
            

        return feats,results