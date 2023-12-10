import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

loss_diff = np.load("data/loss_diff2.npy")
annot_distance = np.load("data/annot_distance2.npy")

L = loss_diff.T

plt.boxplot(L[:,0:10])
plt.show()


A = annot_distance[0,:,:].squeeze()

sim = np.corrcoef(A)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(sim)
plt.colorbar()


A = annot_distance[:,0,:].squeeze()

sim = np.corrcoef(A)

plt.subplot(1,2,2)
plt.imshow(sim)
plt.colorbar()
plt.show()



A = np.mean(annot_distance,axis=1)

print(A.shape)

sim = np.corrcoef(A)
np.fill_diagonal(sim, 0)
plt.imshow(sim)
plt.colorbar()
plt.show()



mdic = {"annot_distance": annot_distance, "loss_diff": loss_diff}

savemat("data/matlab_lesion_data.mat", mdic)