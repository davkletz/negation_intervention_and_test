
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from roberta_interv.IntervRoberta import RobertaForMaskedLM2
from joblib import load

def encode_batch(current_batch, tokenizer, model, device):

    with torch.no_grad():
        encoded_sentence = tokenizer.batch_encode_plus(current_batch,padding=True,  return_tensors="pt").to(device)

        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id)

        tokens_logits = model(**encoded_sentence)

        mask_tokens_logits = tokens_logits['logits'][ mask_tokens_index]

        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices#.tolist()

        predicted_tokens = tokenizer.batch_decode(top_tokens)

    return predicted_tokens


model_name = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model_ref = AutoModelForMaskedLM.from_pretrained(model_name)

a = encode_batch(["I like <mask>", "I like to <mask>", "he is <mask> ugly!"], tokenizer, model_ref, "cpu")




model_2 = RobertaForMaskedLM2.from_pretrained(model_name)


list_n = [40]
P = {}
Ws = {}

for n in list_n:
    P[n] = load(f"{abs_path}/neg_proj_matrix_{model_name}_{n}")
    Ws[n] = load(f"{abs_path}/neg_log_reg_hyperplans_{model_name}_{n}")



P = 0
Ws = 0
cudastring = "cpu"
alpha = 0
direction = 1

model_2.init_alterings(P, Ws, cudastring, alpha, direction)


b = encode_batch(["I like <mask>", "I like <mask>", "he is <mask> ugly!"], tokenizer, model_2, "cpu")

print("\n#######\n")
print(a)

print("\n")
print(b)

print("\n")


print(a == b)
