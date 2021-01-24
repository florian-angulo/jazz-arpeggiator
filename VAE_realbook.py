# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:00:34 2020

@author: louis
"""

import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import data_loader


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
print('using',dev)
device = torch.device(dev)

PITCH_LIST = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
MAIN_QUALITY_LIST = ["maj", "min", "dim"]
EXTRA_QUALITY_LIST = ["N", "maj7", "min7"] 

class VAE(nn.Module):
    N_CHORDS = 16
    N_PITCH = 12
    N_MAIN_QUALITY = 3 # A changer aussi dans data_loader
    N_EXTRA_QUALITY = 3
    SIZE_HIDDEN = 400
    SIZE_LATENT = 40
    
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(self.N_CHORDS * (self.N_PITCH \
                                              * self.N_MAIN_QUALITY \
                                              + self.N_EXTRA_QUALITY), \
                             self.SIZE_HIDDEN)
        self.fc21 = nn.Linear(self.SIZE_HIDDEN, self.SIZE_LATENT)
        self.fc22 = nn.Linear(self.SIZE_HIDDEN, self.SIZE_LATENT)
        
        self.fc3 = nn.Linear(self.SIZE_LATENT, self.SIZE_HIDDEN)
        self.fc4 = nn.Linear(self.SIZE_HIDDEN, self.N_CHORDS * \
                                               (self.N_PITCH \
                                                * self.N_MAIN_QUALITY \
                                                + self.N_EXTRA_QUALITY))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        soft = nn.Sigmoid()
        return soft(self.fc4(h3).view(-1, self.N_CHORDS, self.N_PITCH \
                                      * self.N_MAIN_QUALITY \
                                      + self.N_EXTRA_QUALITY))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.N_CHORDS \
                                        * (self.N_PITCH \
                                           * self.N_MAIN_QUALITY \
                                           + self.N_EXTRA_QUALITY)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_anneal_function(anneal_function, step, k, x0):
    """ Beta update function
        
        Parameters
        ----------
        anneal_function : string
            What type of update (logisitc or linear)
        step : int
            Which step of the training
        k : float
            Coefficient of the logistic function
        x0 : float
            Delay of the logistic function or slope of the linear function

        Returns
        -------
        beta : float
            Weight of the KL divergence in the loss function 

        """
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 16*(12*3+3)), \
                                 x.view(-1, 16*(12*3+3)), reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

step = 0

def train():
    global step
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(realbook_dataset):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        beta = kl_anneal_function('linear', step, 1, 10*len(realbook_dataset))
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), Nchunks,
                100. * batch_idx * len(data) / Nchunks,
                loss.item() / len(data)))
        step += 1
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / Nchunks))


epochs = 20
batch_size = 128
log_interval = 100

realbook_dataset = data_loader.import_dataset()
Nchunks = len(realbook_dataset)
realbook_dataset = torch.split(realbook_dataset, batch_size, 0)
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', dest='load', action='store_true')
    parser.add_argument('--no-load', dest='load', action='store_false')
    parser.set_defaults(LOAD=False)
    LOAD = parser.parse_args().load

    if LOAD:
        model.load_state_dict(torch.load("./model_realbook.pt"))
    else:
        for epoch in range(1, epochs + 1):
            train()
        torch.save(model.state_dict(), "./model_realbook.pt")
        with torch.no_grad():
            inp = realbook_dataset[0].to(device)
            inp = inp[0:3]
            out, mu, logvar = model(inp)
            inp = inp.cpu().numpy()
            out = out.cpu().numpy()
