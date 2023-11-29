import itertools
import time
import sys
sys.path.append("./groups")
from loss import *
from util import *
from models.generator import *
from models.discriminator import *
from models.basic import *
from dataset import *
from torch.utils.data import ConcatDataset
import torch.optim as optim
from torchmetrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure,MetricCollection
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
class CycleGANsformer(nn.Module):
    def __init__(self):
        super().__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        torch.cuda.set_device(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
        self.discX = TDiscriminator().to(device)
        self.device=device
        self.genX2Y = TGenerator().to(device)
        self.state_dict = {
            'self.discX': self.discX.state_dict(),
            'self.genX2Y': self.genX2Y.state_dict()
        }
        
        self.opt_discX = optim.AdamW(
            self.discX.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )

        self.opt_genX2Y = optim.AdamW(
            self.genX2Y.parameters(),
            lr=5e-5,
            weight_decay=1e-5,
        )
        self.opt_genX2Y_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt_genX2Y, mode='min', factor=0.5, patience=10, verbose=True)#optim.lr_scheduler.StepLR(self.opt_genX2Y, step_size=20, gamma=0.5)
        self.opt_discX_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt_discX, mode='min', factor=0.5, patience=10, verbose=True)
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.GradientDifferenceLoss = GDL()
        self.HingeLoss = HingeLoss()
        self.Perceptual_loss = Perceptual_loss134()
        self.deblur_percept = Perceptual_loss()
        self.batch_size=2
    def fit(self, dataset, epochs=100):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        # testloader=DataLoader(ImageDatasetLoader(mode="test",transformers=transforms_ ), batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        evlloader=DataLoader(ImageDatasetLoader(mode="evl",transformers=transforms_ ), batch_size=self.batch_size, shuffle=False, num_workers=4)

        
        print("Start the training!")
        step = 0
        for epoch in range(epochs):
            for idx, (x_img, y_img) in enumerate(loader):
                step+=1
                y_img1=y_img.type(torch.cuda.FloatTensor).cuda(non_blocking=True)
                x_img1=x_img.type(torch.cuda.FloatTensor).cuda(non_blocking=True)
                self.genX2Y.train()
                self.discX.train()
#                 assert fake_x.size() == y_img1.size() , f"fake_imgs.size(): {fake_x.size()} real_imgs.size(): {y_img1.size()}"
#######################################################################
######################################################################                
                self.opt_genX2Y.zero_grad()
                fake_y1 = self.genX2Y(y_img1, x_img1)
                D_Y_real1, segmap1 = self.discX(fake_y1, y_img1)
                # D_Y_real1, segmap1 = self.discX(fake_y1[2], y_img1)
            
                generator_loss = self.HingeLoss(D_Y_real1,segmap1,real = False)
                
                fake_y2 = self.genX2Y(y_img1, x_img1)
                perceptual_loss = self.Perceptual_loss(fake_y2,y_img1)
                # perceptual_loss = self.Perceptual_loss(fake_y2[2],y_img1)
                
                cycle_X = self.genX2Y(y_img1, x_img1)
                deblur_percept_loss =  self.deblur_percept(cycle_X,x_img1)*1 + self.GradientDifferenceLoss(cycle_X,x_img1)*1 
                
                G_loss = generator_loss*1 + perceptual_loss*0.15 + deblur_percept_loss *0.25   #+ GDL_loss *5
                   
                G_loss.backward(retain_graph=True)
                self.opt_genX2Y.step()

##################################################                
                self.opt_discX.zero_grad()
                D_X_real,seg_map2 = self.discX(x_img1,y_img1)

                fake_x = self.genX2Y(y_img1, x_img1)
                D_X_fake,seg_map3 = self.discX(fake_x,y_img1)
                # D_X_fake,seg_map3 = self.discX(fake_x[2],y_img1)
                D_real_loss = self.HingeLoss(D_X_real,seg_map2)
                D_fake_loss = self.HingeLoss(D_X_fake,seg_map3,real = False)
                D_X_loss = D_real_loss*1 + D_fake_loss*1
                D_X_loss.backward(retain_graph=True)
                self.opt_discX.step()
##########################################################################
##########################################################################
            self.opt_genX2Y_scheduler.step(G_loss)
            # self.opt_genX2Y_scheduler.step()
            self.opt_discX_scheduler.step(D_X_loss)
        

        
        
            dir_checkpoint ="./0429lite_checkpoint/"
            torch.save(self.state_dict, dir_checkpoint + f'0429lite_epoch{epoch}.pth')
            print(f'[Epoch {epoch+1}/{epochs}]')
            print(f'[G loss: {G_loss.item()} | generator_loss: {generator_loss.item()} |perceptual_loss: {perceptual_loss.item()} |deblur_percept_loss: {deblur_percept_loss.item()}]')
            print(f'[D loss: { D_X_loss.item()} ]')

            if (epoch) % 2 == 0:
                self.genX2Y.eval()
                with torch.no_grad():
                    ans_ssim = 0
                    ans_psnr_toB = 0
                    ans_psnr_toA = 0
                    for idx, (x_img, y_img) in enumerate(evlloader):
                        y_img1=y_img.type(torch.cuda.FloatTensor).cuda(non_blocking=True)
                        x_img1=x_img.type(torch.cuda.FloatTensor).cuda(non_blocking=True)
                        psnrA = PeakSignalNoiseRatio(data_range=torch.max(x_img1)-torch.min(x_img1)).cuda()
                        psnrB = PeakSignalNoiseRatio(data_range=torch.max(y_img1)-torch.min(y_img1)).cuda()
                        ssim = StructuralSimilarityIndexMeasure().cuda()
                        fake_B = self.genX2Y(y_img1, x_img1)
                        # fake_B = self.deblur(fake_B)

                        ans_ssim +=ssim(fake_B,x_img1)
                        ans_psnr_toB +=psnrB(fake_B,y_img1)
                        ans_psnr_toA +=psnrA(fake_B,x_img1)

                        if idx == len(evlloader)-1:

                            nrows = x_img1.size(0)
                            real_A = make_grid(x_img1, nrow=nrows, normalize=True)
                            fake_B = make_grid(fake_B, nrow=nrows, normalize=True)
                            real_B = make_grid(y_img1, nrow=nrows, normalize=True)

                            image_grid = torch.cat((real_A, fake_B, real_B), 1).cpu().permute(1, 2, 0)

                            plt.figure(figsize=(1.5*nrows, 1.5*3))
                            plt.imshow(image_grid)
                            plt.axis('off')
                            plt.savefig(f'./0429lite_checkpoint/plot/{epoch}.png')
                            plt.show()
                            plt.close(image_grid)



                    print(f"Epoch {epoch+1} | ssim : {ans_ssim/len(evlloader)}")
                    print(f"Epoch {epoch+1} | psnr_toB : {ans_psnr_toB/len(evlloader)}")
                    print(f"Epoch {epoch+1} | psnr_toA : {ans_psnr_toA/len(evlloader)}") 

                    
                    cbct_count=0
                    path="./0429/"+str(epoch)+"/"
                    if not os.path.exists(path):
                        os.mkdir(path)
                    cbct_data=[]
                    for i in range(len(data)):
                        ds=pydicom.dcmread(data[i][0],force=True)
                        cbct_data.append(ds)
                    cbctdata=cbct_data[24600:24704]


                    start_time = time.time()
                    for idx, (a_img, b_img) in enumerate(evlloader):

                        self.genX2Y.eval()
                        a=a_img.type(torch.cuda.FloatTensor)
                        b=b_img.type(torch.cuda.FloatTensor)

                        fake_B = self.genX2Y(b,a).detach()
                        b = fake_B.cpu().numpy()
                        c =b.copy()
                        for s in range(len(a_img)):
                            t=s+idx*self.batch_size
                            write_dicom(cbctdata[t],c[s],path,t)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print("Execution time:", execution_time, "seconds")
if __name__ == "__main__":
    # Create an instance of CycleGANsformer
    cg = CycleGANsformer()

    # Create instances of ImageDatasetLoader for seta and setb
    seta = ImageDatasetLoader(mode="train1", transformers=transforms_)
    setb = ImageDatasetLoader(mode="train", transformers=data_transform)

    # Concatenate the datasets into trainset
    trainset = ConcatDataset([seta, setb])

    # Train the CycleGAN model using seta
    cg.fit(seta)