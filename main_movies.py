from scipy.io import loadmat, savemat
import numpy as np
import model as net
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn



# Hyperparameters
batch_size = 10
sequence_length = 50
num_iter = 20
learning_rate = 0.00001
epochs = 100


ntokens = 200  # fMRI 'vocabulary' size (e.g., number of distinct brain regions or features)
emsize = 2000    # embedding dimension
nhid = 500      # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2     # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2       # the number of heads in the multiheadattention models
out_size = 1836 #size of the output to be predicted
dropout = 0.2   # the dropout value

model = net.FMRI_Transformer(ntokens, emsize, nhead, nhid, nlayers, out_size, dropout)

#expected input size [src_seq_length, batch_size, ntoken]



# open data
data = loadmat('/N/project/networkRNNs/schaefer200_HCP7t_movie_rest_struct.mat')


it = data['HCP_7t_movie_rest']

scan = 0
movies_path = "movies/movie_gray_"+str(scan+1)+".npy"
movie_gray = np.load(movies_path)

for s in range(129):
    subject = s
    TT = it[0,subject]

    ts = TT['movie'][0,scan][0]  #time by nodes

    movie = movie_gray.reshape(movie_gray.shape[0], -1)
    

    if s == 0:
        ts_stack = ts
        movie_stack = movie
    else:
        ts_stack = np.concatenate((ts_stack,ts), axis=0)
        movie_stack = np.concatenate((movie_stack,movie),axis = 0)
    
    



print("feature shape: ",ts_stack.shape)
print("target shape: ",movie_stack.shape)










# Create TensorDataset
#dataset = TensorDataset(torch.from_numpy(ts_stack), torch.from_numpy(movie_stack))
#train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

all_inputs = torch.from_numpy(ts_stack)
all_targets = torch.from_numpy(movie_stack)






# Loss function
criterion = nn.MSELoss()  # Replace with the appropriate loss function for your task

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    for j in range(num_iter):
        #create batches
        inputs = torch.zeros(sequence_length,batch_size,ntokens)
        targets = torch.zeros(sequence_length,batch_size,out_size)
        for i in range(batch_size):
            start = np.random.randint(0,high= all_inputs.shape[0]-sequence_length)
            stop = start+sequence_length
            inputs[:,i,:] = all_inputs[start:stop,:]
            targets[:,i,:] = all_targets[start:stop,:]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs[-1,:,:], targets[-1,:,:])

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        total_loss += loss.item()

    # Print average loss for the epoch
    print("Epoch: ", epoch," Loss: ", total_loss/num_iter)

