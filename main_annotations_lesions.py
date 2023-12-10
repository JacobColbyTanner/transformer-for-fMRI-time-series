
from scipy.io import loadmat, savemat
import numpy as np
import model as net
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn



# Hyperparameters
#scan = 0
use_features = True
batch_size = 1
sequence_length = 25
learning_rate = 0.00001
epochs = 10


ntokens = 200  # fMRI 'vocabulary' size (e.g., number of distinct brain regions or features)
emsize = 2000    # embedding dimension
nhid = 2000      # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 5    # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4       # the number of heads in the multiheadattention models
if use_features == True:
    out_size = 45
else:
    out_size = 859 #size of the output to be predicted
dropout = 0.2   # the dropout value

model = net.FMRI_Transformer(ntokens, emsize, nhead, nhid, nlayers, out_size, dropout)
model.load_state_dict(torch.load('models/transformer_model_annotations2.pth'))

#expected input size [src_seq_length, batch_size, ntoken]



# open data


data = loadmat('/Users/jacobtanner/Brain networks lab Dropbox/Jacob Tanner/jacobcolbytanner/schaefer200_HCP7t_movie_rest_struct.mat')


it = data['HCP_7t_movie_rest']



path = "/Users/jacobtanner/Brain networks lab Dropbox/bnbl_main/data/hcp_7tmovi/annotations/annotations.mat"
A_data = loadmat(path)


features = np.load("data/features.npy", allow_pickle = True).item()

#print(features[0])

def get_batch(brain_data, target_data, sequence_length, batch_size,ntokens,out_size):
    global use_features

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


    inputs = torch.zeros(sequence_length,batch_size,ntokens)
    targets = torch.zeros(sequence_length,batch_size,out_size)
    for i in range(batch_size):
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

annot_distance = np.zeros((ntokens,epochs,out_size))
loss_diff = np.zeros((ntokens,epochs))

import time

for neuron in range(ntokens):
    print("Neuron:  ", neuron)
    start_time = time.time()
    for epoch in range(epochs):
        #print("  Epoch: ",epoch)
        model.eval()  # Set the model to eval mode
        total_loss = 0

        #create batches
        if use_features == True:
            inputs, targets = get_batch(it, features, sequence_length, batch_size,ntokens,out_size)
        else:
            inputs, targets = get_batch(it, A_data, sequence_length, batch_size,ntokens,out_size)

        # Forward pass
        outputs = model(inputs)

        inputs_lesion = inputs
        inputs_lesion[:,:,neuron] = 0
        
        outputs_lesion = model(inputs_lesion)

   
        annot_distance[neuron,epoch,:] = np.absolute(outputs[-1,:,:].squeeze().detach().numpy()-outputs_lesion[-1,:,:].squeeze().detach().numpy())
        if use_features == True:
            loss_real = criterion2(outputs[-1,:,:], targets[-1,:,:])
            loss_lesion = criterion2(outputs_lesion[-1,:,:], targets[-1,:,:])

        else: 
            loss_real = weighted_binary_cross_entropy(outputs[-1,:,:], targets[-1,:,:], weights=class_weights)
            loss_lesion = weighted_binary_cross_entropy(outputs_lesion[-1,:,:], targets[-1,:,:], weights=class_weights)
        
        loss_diff[neuron,epoch] = loss_lesion-loss_real

    time_it_took = time.time()-start_time
    print("Time: ", time_it_took)


np.save("data/loss_diff2.npy",loss_diff)
np.save("data/annot_distance2.npy",annot_distance)

