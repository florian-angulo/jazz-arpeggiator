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
from torchvision import datasets, transforms
from torchvision.utils import save_image
import data_loader
import sample_to_chords as s2c
import itertools
import matplotlib.pyplot as plt

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

        self.fc1 = nn.Linear(self.N_CHORDS * (self.N_PITCH * self.N_MAIN_QUALITY + self.N_EXTRA_QUALITY), self.SIZE_HIDDEN)
        self.fc21 = nn.Linear(self.SIZE_HIDDEN, self.SIZE_LATENT)
        self.fc22 = nn.Linear(self.SIZE_HIDDEN, self.SIZE_LATENT)
        
        self.fc3 = nn.Linear(self.SIZE_LATENT, self.SIZE_HIDDEN)
        self.fc4 = nn.Linear(self.SIZE_HIDDEN, self.N_CHORDS * (self.N_PITCH * self.N_MAIN_QUALITY + self.N_EXTRA_QUALITY))

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
        return soft(self.fc4(h3).view(-1, self.N_CHORDS, self.N_PITCH * self.N_MAIN_QUALITY + self.N_EXTRA_QUALITY))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.N_CHORDS*(self.N_PITCH * self.N_MAIN_QUALITY + self.N_EXTRA_QUALITY)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 16*(12*3+3)), x.view(-1, 16*(12*3+3)), reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta*KLD


def train(epoch):
    model.train()
    train_loss = 0
    beta = epoch/epochs
    for batch_idx, data in enumerate(realbook_dataset):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), Nchunks,
                100. * batch_idx * len(data) / Nchunks,
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / Nchunks))

global conf_matrix
global cm
conf_matrix = np.zeros((108,108))
qal = ['','m','dim']
classes = ''
# for pitch in PITCH_LIST:
#     classes.append([pitch]*3+qal)
# classes.append(['']*72)
 
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    
def test():
    # model.eval()
    test_loss = 0
    # beta = epoch/epochs
    with torch.no_grad():
        for batch_idx, data in enumerate(realbook_dataset):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # test_loss += loss_function(recon_batch, data, mu, logvar).item()
            data  = data.numpy()
            recon = recon_batch.numpy()
            ids = np.transpose(np.append(np.array(range(0,36)),[0,36,72]))
            
            for i in range(len(data)):
                for j in range(len(data[0])):
                    data_id = int(np.dot(data[i,j],ids))
                    # print(data_id)
                    rec1 = recon[i,j,0:36]
                    rec2 = recon[i,j,36:]
                    recon_id = np.argmax(rec1) + np.argmax(rec2)*36
                    # print(recon_id)  
                    
                    conf_matrix[recon_id,data_id] +=1
    plot_confusion_matrix(conf_matrix, classes,True)
    # plot_confusion_matrix(conf_matrix[0:36,0:36], classes,True)
    #         if i == 0:
    #             n = min(data.size(0), 8)
    #             comparison = torch.cat([data[:n],
    #                                   recon_batch.view(batch_size, 1, 28, 28)[:n]])
    #             save_image(comparison.cpu(),
    #                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    # test_loss /= len(mnist_testset.dataset)
    # print('====> Test set loss: {:.4f}'.format(test_loss))


epochs = 5
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
            train(epoch)
            # test(epoch)
        torch.save(model.state_dict(), "./model_realbook.pt")
        with torch.no_grad():
            inp = realbook_dataset[0]
            inp = inp[0:3]
            out,mu,logvar = model(inp) 
            inp = inp.numpy()
            out = out.numpy()
    # with torch.no_grad():
    #     sample = torch.randn(1, 40).to(device)
    #     sample = model.decode(sample).cpu()
    #     sample = sample.numpy()

    #     print(s2c.sample_to_chords(sample))
