import numpy as np
from get_data.get_data import get_data
from train_classifiers.debias import debias
from rlace.rlace import rlace_proj
import torch
import sys
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression


num_iters = int(sys.argv[1])

array_neg, array_pos = get_data()

print(array_neg.shape)

print(array_pos.shape)


labs = [0 for k in range(len(array_neg))]
labs.extend([1 for k in range(len(array_pos))])
arrays = np.concatenate((array_neg, array_pos), axis=0)

labs_np = np.array(labs)
#print(arrays.shape)
#quit()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Working on {device}")

clf_type = sys.argv[2]

if clf_type == "perceptron":
    clf = Perceptron
elif clf_type == "logistic":
    clf = LogisticRegression
elif clf_type == "sgd":
    clf = SGDClassifier

INLP = True

if INLP:
    debias(arrays,labs_np, num_iters, clf, 1024)

else:
    rlace_proj(arrays, labs_np, device, num_iters, clf_type)




