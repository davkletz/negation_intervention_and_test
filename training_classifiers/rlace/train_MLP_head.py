import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

from tests.get_data import get_data

from mlp_head.mlp_head import MLP_head

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#device = "cpu"


def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 1   # number of epochs to run
    batch_size = 1  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    print(f"1 {torch.cuda.memory_allocated(device)}")

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                #print(f"2 {torch.cuda.memory_allocated(device)}")
                #print(f"Epoch {epoch},  batch {start}")
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)

                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                #print(f"3 {torch.cuda.memory_allocated(device)}")
                #print(f"MOS")
                loss.backward()
                #print(f"4 {torch.cuda.memory_allocated(device)}")
                # update weights
                optimizer.step()
                # print progress

                
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc




X, y = get_data()

X_torch, y_torch = torch.from_numpy(X), torch.from_numpy(y) # ne pas mettre sur GPU
X_torch, y_torch = X_torch, y_torch.reshape(-1, 1)
X_torch, y_torch = X_torch.float(), y_torch.float()

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, train_size=0.7, shuffle=True)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)


cv_scores_deep = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = MLP_head().to(device)
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (deep): %.2f" % acc)
    cv_scores_deep.append(acc)



deep_acc = np.mean(cv_scores_deep)
deep_std = np.std(cv_scores_deep)

print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))






