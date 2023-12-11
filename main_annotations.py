
from scipy.io import loadmat, savemat
import numpy as np
import model as net
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

# Hyperparameters
#scan = 0
use_features = True
supercomputer = True
batch_size = 300
sequence_length = 5
num_iter = 20
learning_rate = 0.00001
epochs = 1000


ntokens = 200  # fMRI 'vocabulary' size (e.g., number of distinct brain regions or features)
emsize = 2000    # embedding dimension
nhid = 2000      # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 5     # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 20     # the number of heads in the multiheadattention models
if use_features == True:
    out_size = 45
else:
    out_size = 859 #size of the output to be predicted
dropout = 0.2   # the dropout value

model = net.FMRI_Transformer(ntokens, emsize, nhead, nhid, nlayers, out_size, dropout)

#expected input size [src_seq_length, batch_size, ntoken]


# Weight for each class

class_weights = torch.tensor([0.04, 0.96])  

# Custom BCE Loss with weights
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    
    return torch.neg(torch.mean(loss))


# open data
if supercomputer == True:
    data = loadmat('/N/project/networkRNNs/schaefer200_HCP7t_movie_rest_struct.mat')
    path = "/N/project/networkRNNs/HCP_movie_stimulus/annotations.mat"
    features = np.load("/N/project/networkRNNs/HCP_movie_stimulus/features.npy", allow_pickle = True).item()
else:
    data = loadmat('/Users/jacobtanner/Brain networks lab Dropbox/Jacob Tanner/jacobcolbytanner/schaefer200_HCP7t_movie_rest_struct.mat')
    path = "/Users/jacobtanner/Brain networks lab Dropbox/bnbl_main/data/hcp_7tmovi/annotations/annotations.mat"
    features = np.load("/Users/jacobtanner/Desktop/transformer-for-fMRI-time-series_local/data/features.npy", allow_pickle = True).item()
it = data['HCP_7t_movie_rest']




A_data = loadmat(path)



print(features[0])

def get_batch(brain_data, target_data, sequence_length, batch_size,ntokens,out_size):
    global use_features



    inputs = torch.zeros(sequence_length,batch_size,ntokens)
    targets = torch.zeros(sequence_length,batch_size,out_size)
    for i in range(batch_size):
        if use_features == True:
            subject = np.random.randint(0,high=129)
            scan = np.random.randint(0,high=3)
            
            TT = brain_data[0,subject]
            ts = torch.from_numpy(TT['movie'][0,scan][0])  #time by nodes
            
            A = torch.from_numpy(target_data[scan].T)
            #print("feature shape: ", A.shape)
        else:

            subject = np.random.randint(0,high=129)
            scan = np.random.randint(0,high=3)
            annot = target_data["feature_matrix"][0,scan]
            TT = brain_data[0,subject]
            ts = torch.from_numpy(TT['movie'][0,scan][0])  #time by nodes
            A = torch.from_numpy(annot.reshape(annot.shape[0], -1))

        start = np.random.randint(0,high= ts.shape[0]-sequence_length)
        stop = start+sequence_length
        inputs[:,i,:] = ts[start:stop,:]
        targets[:,i,:] = A[start:stop,:]

    return inputs, targets


# Create TensorDataset
#dataset = TensorDataset(torch.from_numpy(ts_stack), torch.from_numpy(movie_stack))
#train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#all_inputs = torch.from_numpy(ts_stack)
#all_targets = torch.from_numpy(annotation_stack)






# Loss function
criterion = nn.BCELoss()
criterion2 = nn.MSELoss()  

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    start_time = time.time()
    r = []
    for j in range(num_iter):
        #create batches
        if use_features == True:
            inputs, targets = get_batch(it, features, sequence_length, batch_size,ntokens,out_size)
        else:
            inputs, targets = get_batch(it, A_data, sequence_length, batch_size,ntokens,out_size)


        # Forward pass
        outputs = model(inputs)
        #loss = criterion(outputs[-1,:,:], targets[-1,:,:])
        if use_features == True:
            loss = criterion2(outputs[-1,:,:], targets[-1,:,:])
        else: 
            loss = weighted_binary_cross_entropy(outputs[-1,:,:], targets[-1,:,:], weights=class_weights)

        corr = np.corrcoef(outputs[-1,0,:].detach().numpy(),targets[-1,0,:].detach().numpy())
        r.append(corr[0,1])
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        total_loss += loss.item()
    #plt.plot(outputs[-1,0,:].detach().numpy())
    #plt.plot(targets[-1,0,:].detach().numpy())
    #plt.show()

    

    end_time = time.time()-start_time
    # Print average loss for the epoch
    print("Epoch: ", epoch," Loss: ", total_loss/num_iter,"r: ",np.nanmean(r), "time: ", end_time, flush=True)

if supercomputer == True:
    torch.save(net.state_dict(), '/N/project/networkRNNs/model_annotations2.pth')
else:
    torch.save(model.state_dict(), 'models/transformer_model_annotations2.pth')
