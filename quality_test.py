import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from quality_tests.perplexity import compute_perplexity
from quality_tests.neg_verbs_perplexity import compute_perplexity_verbs_neg
import torch
import sys
import numpy as np
from random import random, seed
from roberta_interv.IntervRoberta import RobertaForMaskedLM2
from transformers import AutoTokenizer, AutoModelForMaskedLM


clf_type = sys.argv[1]
n_val = int(sys.argv[2])
alpha = float(sys.argv[3])
direction = int(sys.argv[4])

try :
    quality_test = sys.argv[5]
    if quality_test in ["0", "sentences", "all"]:
        quality_test = "sentences"
    else:
        quality_test = "verbs"

except:
    quality_test = "sentences"

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
    #model_name = "roberta-large"
    if model_name == "InterRoberta":
        model_interv = RobertaForMaskedLM2.from_pretrained("roberta-large")
        model_interv.init_alterings(P, P_row, Ws, alpha, direction)
    else:
        model_interv = AutoModelForMaskedLM.from_pretrained("roberta-large")
        n_val = None

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")


    #model_interv.to(device)

    if quality_test == "verbs":
        pplx = compute_perplexity_verbs_neg(model_interv, tokenizer, n_P = n_val)

    else:
        pplx = compute_perplexity(model_interv, tokenizer, n_P = n_val)


    #print(f"\nn: {n_val}, alpha: {alpha}, direction: {direction}")
    #print(f"Perplexity: {pplx}")

    print(f"\n {n_val} &  {alpha} & direction: {direction} & {round(pplx[0]*100)/100} & {round(pplx[1]*100)/100} ")