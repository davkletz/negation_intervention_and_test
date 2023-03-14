import numpy as np
from get_data.get_data import get_data
from train_classifiers.debias import debias

array_neg, array_pos = get_data()


labs = [0 for k in range(len(array_neg))]
labs.extend([1 for k in range(len(array_pos))])
arrays = np.concatenate((array_neg, array_pos), axis=0)

print(arrays.shape)
quit()


debias(array_neg, array_pos)




