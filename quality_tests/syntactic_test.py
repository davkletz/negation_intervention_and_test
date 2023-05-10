from joblib import load
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
from roberta_interv.IntervRoberta import RobertaForMaskedLM2
import numpy as np





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


def check_gerundive(predictions):

    for pred_available in predictions:
        if not pred_available[-3:] == "ing":
            return False

    return True


def get_all_gerundives_in_distrib(tokenizer):
    vocab_size = tokenizer.vocab_size

    gerundives = []
    for k in range(vocab_size):
        if tokenizer.decode(k)[-3:] == "ing":
            gerundives.append(k)
    return gerundives




def get_verbs_from_ids(tokenizer, ids):
    verbs = []
    for id_available in ids:
        verbs.append(tokenizer.decode(id_available))
    return verbs


def get_gerundives_total_weight(list_gerundives, distrib ):

    tot = 0
    for gerundive in list_gerundives:
        tot += distrib[gerundive]

    return tot




def test_gerundives(sentences, model, tokenizer, mask_id, n_val = None):

    for sentence_available in sentences:

        current_sequence = tokenizer(sentence_available, return_tensors='pt')
        mask_index  = current_sequence['input_ids'][0] == mask_id

        with torch.no_grad():

            if n_val == None:
                output = model(**current_sequence, return_dict=True)
            else:
                output = model(**current_sequence, return_dict=True, n_P=n_val)

            score_token = output["logits"][0, mask_index]


            probas = torch.nn.functional.softmax(score_token, dim=-1)

            total_weight = get_gerundives_total_weight(list_gerundives, probas)



if __name__ == "__main__":


    arg_idx = 0
    model_name = sys.argv[arg_idx]

    if model_name == "InterRoberta":

        model, tokenizer_name, n_val = init_Inter_Roberta()



    else:
        arg_idx += 1
        size = sys.argv[arg_idx]

        if  model_name in ["rob", "roberta", "ROB"]:
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
    list_gerundives = get_all_gerundives_in_distrib


    path = ""


    sentences = load(path)







