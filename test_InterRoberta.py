
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from roberta_interv.IntervRoberta import RobertaForMaskedLM2
from joblib import load
import sys
import numpy as np

def encode_batch(current_batch, tokenizer, model, device, n_P = None):

    with torch.no_grad():
        encoded_sentence = tokenizer.batch_encode_plus(current_batch,padding=True,  return_tensors="pt").to(device)
        print(encoded_sentence['input_ids'].shape)

        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id)

        if n_P is None:
            tokens_logits = model(**encoded_sentence)
        else:
            tokens_logits = model(**encoded_sentence, n_P = n_P)

        mask_tokens_logits = tokens_logits['logits'][ mask_tokens_index]

        #print(f"mask_tokens_logits : {mask_tokens_logits}")

        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices#.tolist()

        predicted_tokens = tokenizer.batch_decode(top_tokens)

    return predicted_tokens


sentences = ["Jordan is a decorator who likes to sing. He really likes to <mask>.", "Jordan is a decorator who likes to sing. He really doesn't like to <mask>."]

n_val  = int(sys.argv[1])
model_name = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_ref = AutoModelForMaskedLM.from_pretrained(model_name)

a = encode_batch(sentences, tokenizer, model_ref, "cpu")




model_2 = RobertaForMaskedLM2.from_pretrained(model_name)

path = "Output"



list_n = [n_val]
P = {}
Ws = {}
P_row = {}

device  = torch.device("cpu")

for n in list_n:
    P[n] = torch.tensor(np.load(f"{path}/P_{n}.pkl")).float().to(device)
    Ws[n] = torch.tensor(np.load(f"{path}/Ws_{n}.pkl")).float().to(device)
    P_row[n] = torch.tensor(np.load(f"{path}/rowspace_projs_{n}.pkl")).float().to(device)


alpha = 0
direction = 1

model_2.init_alterings(P, P_row, Ws, alpha, direction)

n_P = n_val
b = encode_batch(sentences, tokenizer, model_2, "cpu", n_P = n_P)

print("\n#######\n")
print(a)

print("\n NEUTRAL / ")
print(b)

print("\n")


print(a == b)



model_2 = RobertaForMaskedLM2.from_pretrained(model_name)

path = "Output"

list_n = [n_val]
P = {}
Ws = {}
P_row = {}

for n in list_n:
    P[n] = torch.tensor(np.load(f"{path}/P_{n}.pkl")).float().to(device)
    Ws[n] = torch.tensor(np.load(f"{path}/Ws_{n}.pkl")).float().to(device)
    P_row[n] = torch.tensor(np.load(f"{path}/rowspace_projs_{n}.pkl")).float().to(device)


cudastring = "cpu"
alpha = 1
direction = 1

model_2.init_alterings(P,P_row,  Ws, alpha, direction, )

n_P = n_val
b = encode_batch(sentences, tokenizer, model_2, "cpu", n_P = n_P)


print("\n Positive / ")
print(b)

print("\n")


print(a == b)




model_2 = RobertaForMaskedLM2.from_pretrained(model_name)

path = "Output"

list_n = [n_val]
P = {}
Ws = {}
P_row = {}

for n in list_n:
    P[n] = torch.tensor(np.load(f"{path}/P_{n}.pkl")).float().to(device)
    Ws[n] = torch.tensor(np.load(f"{path}/Ws_{n}.pkl")).float().to(device)
    P_row[n] = torch.tensor(np.load(f"{path}/rowspace_projs_{n}.pkl")).float().to(device)


cudastring = "cpu"

direction = -1

model_2.init_alterings(P, P_row, Ws, alpha, direction)

n_P = n_val
b = encode_batch(sentences, tokenizer, model_2, "cpu", n_P = n_P)


print("\n Negative / ")
print(b)

print("\n")


print(a == b)
