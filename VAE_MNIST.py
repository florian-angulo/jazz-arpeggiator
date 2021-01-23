# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:02:27 2020

@author: louis
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():  
  device = torch.device("cuda:0")
else:  
  device = torch.device("cpu")
  
print('using',device)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 40)
        self.fc22 = nn.Linear(400, 40)
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)
        #return min(12, 12*step/x0)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)

    return BCE + beta*KLD



def train(epoch):
    global step
    model.train()
    train_loss = 0
    #beta = 0.5

    for batch_idx, (data, _) in enumerate(mnist_trainset):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        #beta = kl_anneal_function('linear',step,1,8*len(mnist_trainset))
        beta = kl_anneal_function('linear',step,1,4*len(mnist_trainset))
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        step += 1
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(mnist_trainset.dataset),
                100. * batch_idx / len(mnist_trainset),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(mnist_trainset.dataset)))
    print('====> Epoch: {} Latest beta_train: {:.4f}'.format(
          epoch, beta))
    plt_loss.append(train_loss / len(mnist_trainset.dataset))  # For plot loss



def test(epoch):
    global test_step
    model.eval()
    test_loss = 0
    #beta = 0.5
    
    with torch.no_grad():
        
        for i, (data, _) in enumerate(mnist_testset):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            #beta = kl_anneal_function('linear',test_step,1,
            #                          8*len(mnist_testset))
            beta = kl_anneal_function('linear',test_step,1,
                                      4*len(mnist_testset))
            test_loss += loss_function(recon_batch, data, mu, logvar, beta)  \
                                      .item()
            test_step += 1
                                      
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 28, 28)\
                                      [:n]])
                save_image(comparison.to(device),
                          'results/reconstruction_' + str(epoch) + '.png',
                          nrow=n)

    test_loss /= len(mnist_testset.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Epoch: {} Latest beta_test: {:.4f}'.format(
          epoch, beta))



epochs = 20
batch_size = 64
log_interval = 100

mnist_trainset = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
                   batch_size=batch_size, shuffle=True)

mnist_testset = test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
                  batch_size=batch_size, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=7e-4)
step = 0
test_step = 0



if __name__ == "__main__":
    plt_loss = []
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        
        with torch.no_grad():
            sample = torch.randn(64, 40).to(device)
            sample = model.decode(sample).to(device)
            save_image(sample.view(64, 1, 28, 28),
                        'results/sample_' + str(epoch) + '.png')
    
    plt.plot(np.linspace(1,epochs,epochs).tolist(),plt_loss)
    plt.xticks(np.arange(1, epochs+1, step=1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim((0,epochs+1))
    plt.grid()
    plt.show()
