
from scipy.io import loadmat, savemat
import numpy as np
import model as net
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn



# Hyperparameters
#scan = 0

batch_size = 10
sequence_length = 25
num_iter = 10
learning_rate = 0.00001
epochs = 1000


ntokens = 200  # fMRI 'vocabulary' size (e.g., number of distinct brain regions or features)
emsize = 2000    # embedding dimension
nhid = 500      # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2     # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2       # the number of heads in the multiheadattention models
out_size = 859 #size of the output to be predicted
dropout = 0.2   # the dropout value

model = net.FMRI_Transformer(ntokens, emsize, nhead, nhid, nlayers, out_size, dropout)
# Load weights from a .pth file
model.load_state_dict(torch.load('models/transformer_model_annotations.pth'))


#expected input size [src_seq_length, batch_size, ntoken]



# open data


data = loadmat('/Users/jacobtanner/Brain networks lab Dropbox/Jacob Tanner/jacobcolbytanner/schaefer200_HCP7t_movie_rest_struct.mat')


it = data['HCP_7t_movie_rest']



path = "/Users/jacobtanner/Brain networks lab Dropbox/bnbl_main/data/hcp_7tmovi/annotations/annotations.mat"
A_data = loadmat(path)


def get_batch(brain_data, target_data, sequence_length, batch_size,ntokens,out_size):

    subject = np.random.randint(0,high=129)
    scan = np.random.randint(0,high=3)
    annot = target_data["feature_matrix"][0,scan]
    TT = it[0,subject]
    ts = torch.from_numpy(TT['rest'][0,scan][0])  #time by nodes
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
criterion = nn.MSELoss()  # Replace with the appropriate loss function for your task

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.eval()  # Set the model to training mode
    total_loss = 0

    #create batches

    inputs, targets = get_batch(it, A_data, sequence_length, batch_size,ntokens,out_size)

    with torch.no_grad():
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs[-1,:,:], targets[-1,:,:])
    
        total_loss += loss.item()
        

    # Print average loss for the epoch
    print("Epoch: ", epoch," Loss: ", total_loss/(epoch+1))


