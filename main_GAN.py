# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:58:04 2022

@author: Bayazid
"""
#!/usr/bin/python

import os
import sys
import time
import math
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
from IPython.display import clear_output
import imageio

from music21 import midi
import muspy



#%% DCCGAN Architecture
#############################

# ============    Discriminator    ============
    
class Discriminator(nn.Module):
    def __init__(self, features_d):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(in_channels=features_d, out_channels=features_d*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(in_channels=features_d*2, out_channels=features_d*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(in_channels=features_d*4, out_channels=features_d*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(in_channels=features_d*8, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False),
            
            nn.Sigmoid()
            )
            
    def forward(self, x):
        return self.net(x)
            
            
# ============    Generator    ============

class Generator(nn.Module):
    def __init__(self, noise_chnnl, features_g):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_chnnl, out_channels=features_g*8, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(features_g*8),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(in_channels=features_g*8, out_channels=features_g*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g*4),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(in_channels=features_g*4, out_channels=features_g*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g*2),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(in_channels=features_g*2, out_channels=features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(in_channels=features_g, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            
            nn.Sigmoid()
            )
            
    def forward(self, x):
        return self.net(x)


#%% GPU
#############################

use_cuda = torch.cuda.is_available()
GPU_indx  = 0
device = torch.device(GPU_indx if use_cuda else "cpu")


#%% Dataset
#############################

# Importing dataset
data = muspy.MusicNetDataset('data', download_and_extract=True)
data = data.convert() # converting to music object

# transforming music note dataset to pytorch dataset
trainset = data.to_pytorch_dataset(representation="note") 

# Clipping dataset to fit into GAN architecture
dataset_list = []
for music in trainset:
    dataset_list.append(music[:256,:])
    

dataset_list2 = np.array(dataset_list)
dataset_list2 = dataset_list2.reshape(len(dataset_list2),32, 32)

dataset_list2 = torch.Tensor(dataset_list2)

dataset = TensorDataset(dataset_list2) # converting clipped dataset back into pytorch dataset
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32,drop_last=True, shuffle=True, num_workers = 0)




#%% Hyperparameters
#############################

# training parameters
dlr = 0.0001 # discriminator learning rate
glr = 0.001  # generator learning rate
batch_size = 32
train_epoch = 70

features_d = 16 # input channels for discriminator
features_g = 16 # input channels for generator
noise_chnnl = 128 # noise channels for generator

# Creating  the networks
G = Generator(noise_chnnl, features_g).to(device)
D = Discriminator(features_d).to(device)

# A fixed latent noise vector to show the improvement over the epochs
fixed_noise_chnnl = torch.randn(batch_size, noise_chnnl, 1, 1).to(device)

# Binary Cross Entropy loss
BCE_loss = nn.BCEWithLogitsLoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=glr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=dlr, betas=(0.5, 0.999))

#%% Save folder
#############################

if not os.path.isdir(r'C:\Users\Bayazid\Documents\Monash\ECE4179\Project\python_files\Music_DCGAN_results'):
    os.mkdir(r'C:\Users\Bayazid\Documents\Monash\ECE4179\Project\python_files\Music_DCGAN_results')
    
test_images_log = []
D_losses = []
G_losses = []

D_out_fake = []
D_out_real = []


#%% Training loop
#############################

for epoch in range(train_epoch):
    for traindata in enumerate(train_loader):
        
        num_iter, music_notes = traindata
        music_notes = music_notes[0]
        music_notes = music_notes[:, None, :, :]
        music_notes = music_notes.to(device)
        
        #the size of the current minibatch
        mini_batch = music_notes.size()[0]
        
        #Creating the "real" and "fake" labels
        label_real = torch.ones(mini_batch, device = device)
        label_fake= torch.zeros(mini_batch, device = device)    
        
        ########### Training Discriminator ############
        
        #Step1: Sampling a latent vector from a normal distribution and passing it through the generator
        #to get a batch of fake images
        latent_noise = torch.randn(mini_batch, noise_chnnl, 1, 1, device = device)
        G_output = G(latent_noise)
        
        #Step2: Passing the minibatch of real images through the Discriminator and calculating
        #the loss against the "real" label
        #Add some noise so the Discriminator cannot tell that the real image's pixel
        #values are decrete
        input_noise = 0.01*torch.randn_like(music_notes)
        D_real_out = D(music_notes+input_noise).squeeze()
        D_real_loss = BCE_loss(D_real_out, label_real)
        D_out_real.append(D_real_out.mean().item())
        
        #Step3: Passing the minibatch of fake images (from the Generator) through the Discriminator and calculating
        #the loss against the "fake" label
        #Adding some noise so the Discriminator cannot tell that the real image's pixel
        #values are decrete
        input_noise = 0.01*torch.randn_like(music_notes)
        D_fake_out = D(G_output.detach() + input_noise).squeeze()
        D_fake_loss = BCE_loss(D_fake_out, label_fake)
        D_out_fake.append(D_fake_out.mean().item())

        #Step4: Adding the two losses together, backpropogating through the discriminator and taking a training step 
        D_train_loss = D_real_loss + D_fake_loss
        
        D.zero_grad()
        D_train_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.item()) #log the discriminator training loss
                
        ########### Training Generator ##############
        
        #Step1: Sampling a latent vector from a normal distribution and passing it through the generator
        #to get a batch of fake images
        latent_noise = torch.randn(mini_batch, noise_chnnl, 1, 1, device = device)
        G_output = G(latent_noise)
        
        #Step3: Passing the minibatch of fake images (from the Generator) through the Discriminator and calculating
        #the loss against the "real" label - the Generator wants the discriminator to think it's outputs are real
        D_result = D(G_output).squeeze()
        G_train_loss = BCE_loss(D_result, label_real)
        
        # Step4: Backpropogating the loss through the discriminator and into the Generator and taking a training step 
        G.zero_grad()
        G_train_loss.backward()
        G_optimizer.step()
        
        # logging the generator training loss
        G_losses.append(G_train_loss.item())
        
        clear_output(True)
        # Printing out the training status
        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, train_epoch, num_iter+1, len(train_loader), D_train_loss, G_train_loss))
                
    # saving both networks
    torch.save(G.state_dict(), r"C:\Users\Bayazid\Documents\Monash\ECE4179\Project\python_files\Music_DCGAN_results\generator_param.pt")
    torch.save(D.state_dict(), r"C:\Users\Bayazid\Documents\Monash\ECE4179\Project\python_files\Music_DCGAN_results\discriminator_param.pt")
    
    # logging the output of the generator given the fixed latent noise vector
    test_fake = (G(fixed_noise_chnnl) + 1)/2
    imgs_np = (torchvision.utils.make_grid(test_fake.cpu().detach(), 4, pad_value = 0.5).numpy().transpose((1, 2, 0))*255).astype(np.uint8)
    test_images_log.append(imgs_np)

    
#%% Visualizing results
#############################

plt.plot(D_losses)
plt.plot(G_losses)

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title("DCGAN Loss")
plt.legend(["Discriminator loss", "Generator loss"])
plt.savefig("DCGAN_loss.jpg")



#%% Generating Note-Based Output
#############################

# Defining function for linear rescaling
def linear_rescale(array, new_min, new_max):
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

# converting generator output back to original batch of notes
test_gen_fake = G(fixed_noise_chnnl)
test_gen_fake = test_gen_fake.squeeze()
test_gen_fake = test_gen_fake.cpu().detach().numpy()

test_gen_fake = test_gen_fake.reshape(len(test_gen_fake),256, 4) # original shape
test_gen = test_gen_fake

# Rescaling according to library documentation specifics
timescale = [i*25 for i in range(256)]
test_gen[:,:,0] = timescale[:]
test_gen[:,:,1] = linear_rescale(test_gen[:,:,1], 50, 100)
test_gen[:,:,2] = linear_rescale(test_gen[:,:,2], 30, 500)
test_gen[:,:,3] = 100

test_gen = test_gen.astype(int)


#%% Generating MIDI
#############################
midi_generated_note = muspy.from_note_representation(test_gen[10,:,:])
muspy.write_midi(r'C:\Users\Bayazid\Documents\Monash\ECE4179\Project\midi_gen\generated_note_5.mid', midi_generated_note)



