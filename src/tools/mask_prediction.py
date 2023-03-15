import torch
from src.tool.alter_representation import alter_representation

def mask_prediction(sentence_available, tokenizer, model, device, conj_act_token_ids):
    encoded_sentence = tokenizer.encode(sentence_available, return_tensors="pt").to(device)
    mask_token_index = torch.where(encoded_sentence == tokenizer.mask_token_id)[1]
    context_vect = model.roberta(encoded_sentence)

    altered_rpz = alter_representation(context_vect)

    token_logits = model.lm_head(altered_rpz)


    mask_token_logits = token_logits[0, mask_token_index, :]
    indice_act, sorted, probas_tok = get_indice_act(mask_token_logits, conj_act_token_ids)
    top_token = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
    predicted_token=tokenizer.decode([top_token[0]])
    return predicted_token, mask_token_logits[0], indice_act, probas_tok




def get_indice_act(token_logits, conj_act_token_ids):

    probas = torch.softmax(token_logits[0], dim=0)

    sorted, indices = torch.sort(probas, descending=True)

    if type(conj_act_token_ids) == int:
        indx = (indices == conj_act_token_ids).nonzero()

    else:

        if conj_act_token_ids[0] == conj_act_token_ids[1]:
            indx = (indices == conj_act_token_ids[0]).nonzero()

        else:
            indx_0 = (indices == conj_act_token_ids[0]).nonzero()
            indx_1 = (indices == conj_act_token_ids[1]).nonzero()
            indx = min(indx_0, indx_1)



    return indx.item(), indices, sorted[indx]









def batch_mask_prediction(batch_sentences, tokenizer, model, device, size_batches = 8):

    all_predictions = []

    nb_batches = len(batch_sentences) // size_batches
    for k in range(nb_batches):
        current_batch = batch_sentences[k*size_batches:(k+1)*size_batches]
        predicted_tokens = encode_batch(current_batch, tokenizer, model, device)
        all_predictions.extend(predicted_tokens)



    if len(batch_sentences) % size_batches != 0:
        last_batch = batch_sentences[nb_batches*size_batches:]
        predicted_tokens = encode_batch(last_batch, tokenizer, model, device)
        all_predictions.extend(predicted_tokens)


    return all_predictions


def make_and_encode_batch(current_batch, tokenizer, model, device, batch_verbs, name_available, profession_available, current_pronouns_maj, complete_check, found):
    current_found = found
    good_pred = 0
    detail_verbs = []

    predictions = encode_batch(current_batch, tokenizer, model, device)

    new_sentence = None

    #print(batch_verbs)


    for i, prediction_available in enumerate(predictions):
        good_verb = batch_verbs[i]

        same = False
        if good_verb == prediction_available:
            same = True
        if "roberta" in model.__class__.__name__.lower() or "cuenbert" in model.__class__.__name__.lower():
            if f"Ġ{good_verb}" == prediction_available:
                same = True

            if f" {good_verb}" == prediction_available:
                same = True
        if "xlnet" in model.__class__.__name__.lower():
            if f" {good_verb}" == prediction_available:
                same = True

            if f"_{good_verb}" == prediction_available:
                same = True

        #print(f"\n {current_batch[i]} : ({prediction_available}) VS ({good_verb}) ; {same}")

        if same:
            detail_verbs.append(good_verb)
            good_pred += 1
            good_dico = {"name_available": name_available, "profession_available": profession_available,
                         "verb": good_verb, "current_pronouns_maj": current_pronouns_maj}

            if not current_found:
                new_sentence = good_dico

                current_found = True
                if not complete_check:
                    break
    return new_sentence, current_found, good_pred, detail_verbs


def encode_batch(current_batch, tokenizer, model, device):

    with torch.no_grad():
        encoded_sentence = tokenizer.batch_encode_plus(current_batch,padding=True,  return_tensors="pt").to(device)
        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id)
        #print(mask_tokens_index)
        tokens_logits = model(**encoded_sentence)
        #print(tokens_logits)
        #print(tokens_logits['logits'].shape)

        mask_tokens_logits = tokens_logits['logits'][ mask_tokens_index]
        #print(mask_tokens_logits.shape)
        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices#.tolist()
        #print(top_tokens)
        predicted_tokens = tokenizer.batch_decode(top_tokens)
        #print(predicted_tokens)

    return predicted_tokens