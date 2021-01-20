import os
import time
import torch
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict, defaultdict
from model import RVAE
from data_loader_RVAE import import_dataset

def main():
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    print_every = 150
    batch_size = 10
    one_hots, len_sentences = import_dataset()
    
    len_sentences = torch.from_numpy(np.array(len_sentences))
    save_model_path = "./training"
    model = RVAE(max(len_sentences))

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)


  
    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    CE = torch.nn.BCELoss(reduction='sum')
    def loss_fn(pred, target, length, mean, logv, anneal_function, step, k, x0):
       
        # cut-off unnecessary padding from target, and flatten

        target = torch.reshape(target[:, 1:,:],(batch_size,-1))
        pred = torch.reshape(pred[:,:-1,:],(batch_size,-1))
       
        # Negative Log Likelihood
        CE_loss = CE(pred,target)/length.float().mean()

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return CE_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    epochs = 1000
    dataset = TensorDataset(one_hots, len_sentences)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
        )
    for epoch in range(epochs):


        tracker = defaultdict(tensor)

        model.train()


        for iteration, (seq_data, length) in enumerate(data_loader):
            
            #print("max length=",max(length))
            batch_size = seq_data.size(0)

            if torch.cuda.is_available():
                seq_data = seq_data.cuda()

            # Forward pass
            recons_data, mean, logv, z = model(seq_data, length)

            # loss calculation
            
            NLL_loss, KL_loss, KL_weight = loss_fn(recons_data, seq_data[:,:max(length)],
                length, mean, logv, 'logistic', step, 2.5e-4, 20000)

            loss = (NLL_loss + KL_weight * KL_loss) / batch_size
           
            # backward + optimization
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # bookkeepeing
            tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)
            tracker['NLL_Loss'] = torch.cat((tracker['NLL_Loss'], (NLL_loss/batch_size).data.view(1, -1)), dim=0)
            tracker['KL_Loss'] = torch.cat((tracker['KL_Loss'], (KL_loss/batch_size).data.view(1, -1)), dim=0)
            
            if iteration % print_every == 0 or iteration+1 == len(data_loader):
                print("Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                      % (iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                      KL_loss.item()/batch_size, KL_weight))

        

        print("Epoch %02d/%i, Mean ELBO %9.4f" % (epoch, epochs, tracker['ELBO'].mean()))



        # save checkpoint
        if epoch % 10 == 9:
          checkpoint_path = os.path.join(save_model_path, "E%i.pt" % epoch)
          torch.save(model.state_dict(), checkpoint_path)
          print("Model saved at %s" % checkpoint_path)
        
        torch.save(tracker, "./tracker.pt")
        print("Tracker saved")
        
if __name__ == '__main__':
    main()