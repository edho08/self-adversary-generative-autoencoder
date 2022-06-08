# Feedforward net MNIST dataset
import torch
import torchvision
import torchvision.transforms.functional as F
#from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import matplotlib.pyplot as plt
import copy 
import numpy as np
from tqdm import tqdm

#device info
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

#hyper parameter
input_size = 28*28
hidden_size = 100
num_epoch = 100
batch_size = 4
learning_rate = 0.001
step = 1000

gamma = 0.2
lambda_k = 0.001
k_t = 0
epsilon = 1e-7

#MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#show
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Conv2d(16,1,kernel_size=3),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(True)
            )
            
        self.encoder_linear = nn.Sequential(
            nn.Linear(24*24, 8),
            nn.LeakyReLU()
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(8, 24*24),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(1,16,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16,1,kernel_size=3),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(True),
            nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.encoder_linear(x.reshape(-1, 24*24))
        x = self.decoder_linear(x).reshape(-1, 1, 24, 24)
        x = self.decoder(x)
        return x
        
class Generator(nn.Module):
    def __init__(self, linear, decoder):
        super(Generator, self).__init__()
        
        self.linear = linear
        self.decoder = decoder
    
    def forward(self, x):
        x = self.linear(x).reshape(-1, 1, 24, 24)
        x = self.decoder(x)
        return x

discriminator = Autoencoder().to(device)
generator = Generator(discriminator.decoder_linear, discriminator.decoder)

Loss = nn.MSELoss()
optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=learning_rate)

#Training Loop
for epoch in range(num_epoch):
  past = copy.deepcopy(discriminator)
  for i, (imgs, _) in enumerate(tqdm(train_loader)):
    #real imgs
    imgs = imgs.to(device)
    
    #randomly generated latent vectors
    r = torch.randn(batch_size, 8).to(device)
    
    #generate fake image
    fake = generator(r)
    
    #zero_grad
    optimizer.zero_grad()
    optimizer_gen.zero_grad()
    
    #discriminator trying to learn the distribution of real data
    ##discriminator trying to reconstruct real image
    y = discriminator(imgs)
    
    #calculate image reconstruction loss
    loss_dis = Loss(y, imgs)
        
    #make a copy
    loss_dis_real = loss_dis.detach()
    
    #discriminator trying to learn the distribution of fake data
    #for computation reason, only consider doing this when k_t > epsilon
    if k_t > epsilon:
      #discriminator trying to reconstruct fake image
      y_fake = discriminator(fake.detach())
      
      #calculate fake image reconturction loss
      loss_dis -= k_t * Loss(y_fake, fake) 
 
    #this is the step where generator learn.
    #discriminator trying to reconstruct the fake image
    y_fake = past(fake)
    
    #calculate fake image reconstruction loss
    loss_gen = Loss(y_fake, fake)
    
    #backward pass
    total_loss = loss_dis + loss_gen
    total_loss.backward()

    #step
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
    optimizer.step()
    optimizer_gen.step()
    
    #total loss
    loss = loss_dis + loss_gen

    #renew self
    past = copy.deepcopy(discriminator)

        
    #update balancing term k_t
    equilibrium = gamma * loss_dis_real.cpu().item() - loss_gen.detach().cpu().item()
    k_t = k_t + lambda_k * (equilibrium)
    k_t = np.clip(k_t, 0, 1)

    if (i+1) % step == 0:
      grid = utils.make_grid(fake)
      show(grid)
      plt.savefig(f"fig/{epoch}_{i+1}.png")
      plt.close()
      
    i+= 1
  
