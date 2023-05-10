import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import sys
from random import random, seed


def get_perplexity_sequence(model,tokenizer, input_texts: str, n_P = None ):
    """ Compute the perplexity on MLM.

    :param input_texts: A string or list of input texts for the encoder.
    :param batch: Batch size
    :return: A value or list of perplexity.
    """

    # run model
    ppl = []

    tokens = tokenizer(input_texts, return_tensors='pt')
    #print(tokens)

    labels = tokens['input_ids']


    #print(tokens)
    for n_seq in range(len(labels)):

        for k in range(len(labels[n_seq])-2):
            current_sequence ={}
            id_hidden_token = tokens['input_ids'][n_seq][k+1]

            current_sequence['input_ids'] = torch.clone(tokens['input_ids'][n_seq][:k+2].unsqueeze(0))
            current_sequence['input_ids'][n_seq][-1] = 103
            current_sequence['attention_mask'] = tokens['attention_mask'][n_seq][:k+2].unsqueeze(0)
            if "token_type_ids" in tokens:
                current_sequence['token_type_ids'] = tokens['token_type_ids'][n_seq][:k+2].unsqueeze(0)

            #print(current_sequence)
            #print("\n\n")
            #print(id_hidden_token)

            with torch.no_grad():

                if n_P == None:
                    output = model(**current_sequence, return_dict=True)
                else:
                    output = model(**current_sequence, return_dict=True, n_P = n_P)

                prediction_scores = output['logits']
                if len(prediction_scores.shape) == 3:
                    score_token = prediction_scores[n_seq, -1]
                else:
                    score_token = prediction_scores[-1]

            probas = torch.nn.functional.softmax(score_token, dim=-1)



            good_proba = -torch.log(probas[id_hidden_token])
            #print(good_proba)
            ppl.append(good_proba.cpu().tolist())


            #print(prediction_scores.shape)
            target = torch.zeros(prediction_scores[-1].shape, dtype=torch.long)
            target[n_seq, id_hidden_token] = float(1)
            #print(target)

            #print(loss(prediction_scores[-1],target ))


    perplexity = sum(ppl) / len(ppl)

    return perplexity





'''a = get_perplexity_sequence(model,tokenizer, "Though coffee is now a global commodity")


print(a)


a = get_perplexity_sequence(model,tokenizer, "Though elephant phone furiously mine big plain")

print(a)



a = get_perplexity_sequence(model,tokenizer, "Though elephant plain")

print(a)
'''



def compute_perplexity(model, tokenizer, n_P = None):
    dataset = load_dataset("ptb_text_only")

    loss = torch.nn.CrossEntropyLoss()

    all_ppl = []

    tot = 0
    for sentence in dataset["train"]["sentence"]:
        if random() > 0.1:
            continue
        text_to_test = sentence
        text_to_test = text_to_test.replace("<unk>", "[UNK]")
        a = get_perplexity_sequence(model, tokenizer, text_to_test, n_P)
        all_ppl.append(a)

        '''if tot% 100 == 0:
            print("\n\n")
            print(tot)
            print(sum(all_ppl) / len(all_ppl))
    '''
        tot += 1

    return sum(all_ppl)/len(all_ppl)



if __name__ == "__main__":
    seed(42)
    model_name = sys.argv[1]
    if model_name in ["bert-base-cased", "bert-large-cased", "roberta-base","roberta-large" ]:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    else:
        if "inter" in model_name:
            if "roberta" in model_name:
                tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            elif "bert" in model_name:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


            model = AutoModelForMaskedLM.from_pretrained(model_name)

    ppl = compute_perplexity(model, tokenizer)

    print(f"perplexity of {model_name} : {ppl}")