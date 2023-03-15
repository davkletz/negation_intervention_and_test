import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from src.tools.build_array import build_masked_sentences
from src.tools.mask_prediction import mask_prediction
from TG_comm import launchFct
import numpy as np
from joblib import load




path_sentences = "/home/dkletz/tmp/pycharm_project_99/2022-23/neg-eval-set/en_neg-eval/Inputs"

model_name = "roberta-large"
list_good_patterns_model = load( f"{path_sentences}/{model_name}/list_good_patterns_mono_{model_name}.joblib")


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
for param in model.parameters():
    param.requires_grad = False


mask_token = tokenizer.mask_token


hypo_sentence_cons = "NOM est MET qui a l'habitude de ACTION. PRON_maj ne MASK vraiment pas souvent."

pronouns = {"She": "she", "He": "he"}

total_sentences = 0

total_repetition = 0

rg_act_tok = []

for pattern in list_good_patterns_model:
    name_available = pattern["name_available"]
    profession_available = pattern["profession_available"]
    verb_available = pattern["verb"]
    current_pronouns_maj = pattern["current_pronouns_maj"]
    current_pronoun = pronouns[current_pronouns_maj]

    current_hypo_sentence = hypo_sentence_cons

    verb_available_id = tokenizer.encode(verb_available)[1]
    verb_available_id = tokenizer.convert_tokens_to_ids([verb_available, f"Ġ{verb_available}"])
    verb_available_id = [k for k in verb_available_id if k != 3]
    if len(verb_available_id) == 1:
        verb_available_id = verb_available_id[0]

    masked_sentence = build_masked_sentences(current_hypo_sentence, name_available, profession_available,
                                             verb_available,
                                             current_pronouns_maj, current_pronoun, mask_token)

    prediction_conj, mask_token_logits, indice_act, probas_tok = mask_prediction(masked_sentence, tokenizer, model,
                                                                                 device, verb_available_id)
    rg_act_tok.append(indice_act)

    sente_pred = masked_sentence.replace(mask_token, prediction_conj)


    if verb_available == prediction_conj or f" {verb_available}" == prediction_conj:
        total_repetition += 1

    total_sentences += 1


print("\n=====================================\n")
print(model_name)
print(total_repetition)
print(total_sentences)
print((round((total_repetition/total_sentences), 3))*100)
print(f"rg token : {np.mean(rg_act_tok)}")