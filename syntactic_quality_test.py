import sys
from roberta_interv.IntervRoberta import RobertaForMaskedLM2
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from joblib import load
from quality_tests.syntactic_test import check_gerundive, get_all_gerundives_in_distrib, get_verbs_from_ids, get_gerundives_total_weight, test_gerundives


def init_Inter_Roberta():
    arg_idx = 0
    arg_idx += 1
    clf_type = sys.argv[arg_idx]
    arg_idx += 1
    n_val = int(sys.argv[arg_idx])
    arg_idx += 1
    alpha = float(sys.argv[arg_idx])
    arg_idx += 1
    direction = int(sys.argv[arg_idx])


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
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    for n in list_n:
        P[n] = torch.tensor(np.load(f"{path}/P_{clf}_{n}", allow_pickle=True)).float().to(device)
        Ws[n] = torch.tensor(np.load(f"{path}/Ws_{clf}_{n}", allow_pickle=True)).float().to(device)
        P_row[n] = torch.tensor(np.load(f"{path}/rowspace_projs_{clf}_{n}", allow_pickle=True)).float().to(device)

    model_interv = RobertaForMaskedLM2.from_pretrained("roberta-large")
    model_interv.init_alterings(P, P_row, Ws, alpha, direction)
    tokenizer_name = "roberta-large"

    return model_interv, tokenizer_name, n_val


arg_idx = 0
model_name = sys.argv[arg_idx]

if model_name == "InterRoberta":

    model, tokenizer_name, n_val = init_Inter_Roberta()



else:
    arg_idx += 1
    size = sys.argv[arg_idx]

    if model_name in ["rob", "roberta", "ROB"]:
        model_name = "roberta-"

    elif model_name in ["bert", "BERT"]:
        model_name = "bert-"

    if size in ["large", "l", "L"]:
        size = "large"
    else:
        size = "base"
    model_name = model_name + size
    tokenizer_name = model_name
    model = AutoModelForMaskedLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
list_gerundives = get_all_gerundives_in_distrib(model, tokenizer, tokenizer.mask_token_id, n_val)

path = f"/data/dkletz/data/sentences_neg_annot_gerund/"

sentences_neg = load(f"{path}/sentences_with_neg.joblib")
sentences_pos = load(f"{path}/sentences_without_neg.joblib")

probas, weights = test_gerundives(sentences_neg, model, tokenizer, tokenizer.mask_token_id, n_val)



