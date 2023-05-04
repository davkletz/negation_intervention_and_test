
from joblib import load
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer



def get_perplexity_sequence(model,tokenizer, input_texts: str, start, n_P = None ):
    """ Compute the perplexity on MLM.

    :param input_texts: A string or list of input texts for the encoder.
    :param batch: Batch size
    :return: A value or list of perplexity.
    """



    tokens = tokenizer(input_texts, return_tensors='pt')
    #print(tokens)

    labels = tokens['input_ids']


    #print(tokens)
    for n_seq in range(len(labels)):


        current_sequence ={}
        id_hidden_token = tokens['input_ids'][n_seq][start]

        current_sequence['input_ids'] = torch.clone(tokens['input_ids'][n_seq][:start+1].unsqueeze(0))
        current_sequence['input_ids'][n_seq][-1] = 103
        current_sequence['attention_mask'] = tokens['attention_mask'][n_seq][:start+1].unsqueeze(0)
        if "token_type_ids" in tokens:
            current_sequence['token_type_ids'] = tokens['token_type_ids'][n_seq][:start+1].unsqueeze(0)


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
        perplexity = good_proba.cpu().tolist()





    return perplexity





first_page = 0
all_sentences_with_neg = load(f"quality_tests/data/sentences_with_neg{first_page}.pkl")
all_sentences_without_neg = load(f"quality_tests/data/sentences_without_neg{first_page}.pkl")



ds = ['with', 'without']

model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForMaskedLM.from_pretrained(model_name)

datasets = [all_sentences_with_neg, all_sentences_without_neg]



def compute_perplexity_verbs_neg(model, tokenizer,datasets = datasets, n_P = None):

    all_ppl = []

    tot = 0
    for i, c_dataset in datasets:
        for sentence in c_dataset:

            #text_to_test = sentence[0].metadata['text']
            phrase = sentence[0]
            index_found = sentence[1]
            start = sentence[2]
            end = sentence[3]


            a = get_perplexity_sequence(model, tokenizer, text_to_test,  start, n_P)
            all_ppl.append(a)

            if tot% 100 == 0:
                print("\n\n")
                print(tot)
                print(sum(all_ppl) / len(all_ppl))

            tot += 1

        print(f"Dataset {ds[i]} : ")
        print(sum(all_ppl)/len(all_ppl))




#compute_perplexity(model, tokenizer, datasets, n_P )