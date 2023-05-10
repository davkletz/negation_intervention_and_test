from joblib import load
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
import numpy as np






def check_gerundive(predictions):

    for pred_available in predictions:
        if not pred_available[-3:] == "ing":
            return 0

    return 1


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

    tot_sentences = 0
    tot_first = 0
    tot_total_weight = 0


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
            tot_total_weight += total_weight

            prediction = torch.argmax(probas)
            prediction = prediction.item()
            token_predicted = tokenizer.decode(prediction)
            tot_first += check_gerundive(token_predicted)

    return tot_first/tot_sentences, tot_total_weight/tot_sentences










