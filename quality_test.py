import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from quality_tests.perplexity import compute_perplexity
import torch
import sys
import numpy as np
from random import random, seed
from roberta_interv.IntervRoberta import RobertaForMaskedLM2
from transformers import AutoTokenizer


clf_type = sys.argv[1]
n_val = int(sys.argv[2])
alpha = float(sys.argv[3])
direction = int(sys.argv[4])

path = "Output"

if clf_type == "sgd":
    clf = "SGD"
elif clf_type == "lr":
    clf = "LogisticRegression"
elif clf_type == "perceptron":
    clf = "Perceptron"


list_n = [n_val]
P = {}
Ws = {}
P_row = {}
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

for n in list_n:
    P[n] = torch.tensor(np.load(f"{path}/P_{clf}_{n}", allow_pickle=True)).float().to(device)
    Ws[n] = torch.tensor(np.load(f"{path}/Ws_{clf}_{n}", allow_pickle=True)).float().to(device)
    P_row[n] = torch.tensor(np.load(f"{path}/rowspace_projs_{clf}_{n}", allow_pickle=True)).float().to(device)


if __name__ == "__main__":

    seed(42)
    model_name = "InterRoberta"


    tokenizer = AutoTokenizer.from_pretrained("roberta-large")


    model_interv = RobertaForMaskedLM2.from_pretrained("roberta-large")
    model_interv.init_alterings(P, P_row, Ws, alpha, direction)
    #model_interv.to(device)


    pplx = compute_perplexity(model_interv, tokenizer, n_P = n_val)


    print(f"Perplexity: {pplx}")