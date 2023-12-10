from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt



it = loadmat("data/annot_data_for_jake.mat")


Ci = it["Ci"][:,3]

#print(Ci)


path = "/Users/jacobtanner/Brain networks lab Dropbox/bnbl_main/data/hcp_7tmovi/annotations/annotations.mat"
A_data = loadmat(path)


features = {}
for scan in range(4):
    print("scan: ",scan)
    annot = A_data["feature_matrix"][0,scan]
    features[scan] = np.zeros((np.max(Ci),annot.shape[0]))
    for i in range(np.max(Ci)):
        idx = Ci == i+1

        features[scan][i,:] = np.mean(annot[:,idx],axis=1)

        #plt.plot(np.mean(annot[:,idx],axis=1))
        #plt.show()


np.save("data/features.npy",features)

