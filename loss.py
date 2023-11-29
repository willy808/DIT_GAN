import torch
import torch.nn.functional as F
from torch import nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchvision
import sys
sys.path.append("./groups")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
class MS_SSIM_Loss2(MS_SSIM):
    def forward(self, img1, img2):
        return 1*( 1 - super(MS_SSIM_Loss2, self).forward(img1, img2) )
MS_SSIM_Loss = MS_SSIM_Loss2(data_range=1.0, size_average=True, channel=1)
class CustomSSIMLoss(SSIM):
    def forward(self, img1, img2):
        return 1 * (1 - super(CustomSSIMLoss, self).forward(img1, img2))

SSIM_Loss = CustomSSIMLoss(data_range=1.0, size_average=True, channel=1)
# import torch.autograd as autograd
class GDL(nn.Module):
    def __init__(self):
        super(GDL, self).__init__()

    def forward(self, pred, target):
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_diff_x = torch.abs(target_dx - pred_dx)
        grad_diff_y = torch.abs(target_dy - pred_dy)
        return grad_diff_x.mean() + grad_diff_y.mean()

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, predictions,seg,real=True):
        # Compute the hinge loss using Swish activation
        loss = 0
        if real ==True:
            for pred in seg:
                loss += torch.mean(nn.ReLU(inplace=True)(1 - pred))
            for preds in predictions:
                loss += torch.mean(nn.ReLU(inplace=True)(1 - preds))
        else:
            for pred in seg:
                loss += torch.mean(nn.ReLU(inplace=True)(1 + pred))
            for preds in predictions:
                loss += torch.mean(nn.ReLU(inplace=True)(1 + preds))
            
        return loss

class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out, self).__init__()
        self.input=nn.Conv2d(in_channels=1 , out_channels=3, kernel_size=1)
        vgg = torchvision.models.vgg19(pretrained=True).to(device) #.cuda()
        vgg.eval()
        vgg_pretrained_features = vgg.features
        #print(vgg_pretrained_features)
        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2): #(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7): #(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12): #(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21): #(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):#(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

 
    def forward(self, X):
        h=self.input(X)
        h_relu1 = self.slice1(h)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Perceptual_loss134(nn.Module):
    def __init__(self):
        super(Perceptual_loss134, self).__init__()
        self.input=nn.Conv2d(in_channels=1 , out_channels=3, kernel_size=1)
        self.vgg = Vgg19_out().to(device)
        
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.weights = [1.0/32, 1.0/4, 1.0/2, 1.0/1, 1.0/32]
        # self.weights = [1.0/2.6, 1.0/16, 1.0/2, 1.0/1, 1.0]    
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # for i in range(len(x_vgg)-2):
        #     loss += self.weights[i] * self.mse(x_vgg[i], y_vgg[i].detach())    
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.L1(x_vgg[i], y_vgg[i].detach())        
        return loss
class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.input=nn.Conv2d(in_channels=1 , out_channels=3, kernel_size=1)
        self.vgg = Vgg19_out().to(device)
        
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.MS_SSIM_Loss = MS_SSIM_Loss2(data_range=1.0, size_average=True, channel=1)
        #self.weights = [1.0/2.6, 1.0/16, 1.0/3.7, 1.0/5.6, 1.0]
        self.weights = [1.0/15, 1.0/9, 1.0/4, 1.0/3, 1.0/1]    
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # loss += self.weights[2] * self.L1(x_vgg[2], y_vgg[2].detach())
        # loss += self.weights[4] * self.L1(x_vgg[4], y_vgg[4].detach())
        # loss += self.weights[3] * self.mse(x_vgg[3], y_vgg[3].detach())
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.L1(x_vgg[i], y_vgg[i].detach()) 
        # for i in range(len(x_vgg)-1):
        #     loss += self.weights[i] * self.MS_SSIM_Loss(x_vgg[i], y_vgg[i].detach())
        
        return loss
