####
#Author: Milena Nedeljkovic (from https://github.com/milenanedeljkovic/thesis)
#Modified by : David Kletz
# This script creates the vectors for the training and test sets
# It uses the RoBERTa model to get contextual embeddings for each verb
# Verbs in and out the subtree of a negation cue are saved in different lists
# The resulting vectors are saved in a .pt file
# The .pt file should be used in the training script

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import spacy_conll
import spacy_stanza
import torch
from transformers import AutoModel, AutoTokenizer
import torch
from datetime import datetime

import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
rec_depth = sys.getrecursionlimit()

def txt_to_conll(text: str, nlp):
    """Input:
    - text: the string we want to parse
    - nlp: stanza parser (initalized in the cell above)

    Output:
    - the dependency trees for each setnence in text,
      concatenated in a .conll format"""

    # text clean-up: we need to eliminate all \n and \t and not have more than one ' ' in a row anywhere
    # we do this by using string.split() method which splits by " ", \n and \t and concatenate all the pieces
    # excess spaces result in a wrong .conll that is undreadable afterwards
    text = " ".join(text.split())

    doc = nlp(text)
    return doc._.conll_str

import conllu
from typing import List

def stanza_to_bert_tokens(phrase, bert_tokenization, tokenizer):
    """
    The function finds the ranges of tokens in tokenizer's tokenization that correspond to stanza's tokenization

    Input:
    - phrase: the phrase parsed using stanza (will be of the form: TokenList<I, am, home, metadata={sent_id: "1", text: "I am home"}>)
    - bert_tokenization: the output of tokenizer(original-phrase-string)['input_ids']
    - tokenizer: the tokenizer that we will use for getting context tensors
    Output:
    - mapping: the map between stanza tok
    enization and model tokenization
    """

    bert_tokens = tokenizer.convert_ids_to_tokens(bert_tokenization)
    token_map: List[tuple(int, int)] = []  # the tuples mapping (left-to-right) the stanza tokens
    # to the range where they are in the RoBERTa tokenization

    i = 1  # to walk through bert_tokens. Starts at 1 because bert_tokens[0] is the bos token
    j = 0  # to walk through the RoBERTa token character by character.
    # This will help if words are weirdly cut and glued together
    for token in phrase:  # this will loop through all stanza tokens
        token = token['form']
        start = i
        while len(token) > 0:

            if bert_tokens[i][j] == "Ġ":  # this signifies the start of a word in RoBERTa in the pre-tokenized phrase
                j += 1

            if len(token) == len(bert_tokens[i][j:]):  # RoBERTa token corresponds exactly to the stanza token
                token_map.append((start, i + 1))
                i += 1
                j = 0
                break  # breaks the while

            elif len(token) < len(bert_tokens[i][j:]):
                token_map.append((start, i + 1))
                j += len(token)
                break

            else:  # so len(token) > len(bert_tokens[i][j:])
                token = token[len(bert_tokens[i][j:]):]
                i += 1
                j = 0

    return token_map


def get_sentences(path: str, tokenizer):
    """
    Input:
    - path: the location of the .conll file containing dependency trees for each phrase of the text we are analysing
    - tokenizer, model: the tokenizer and the model we will use for contextual representation

    Output:
    - a dictionary mapping each verb lemma to its v- and v+ representations (look into the code for a more detailed explanation)
    """

    # we find negation cues by depth-first search through the tree
    # jump below the function definition


    list_sentences = []
    def depth_search(root, current_verb: str, current_index: int, in_clause: bool, n: int = 1) -> dict:
        """Input:
        - root: the (sub)tree in which we are looking for negation
        - current_verb: if we encounter negation, this is the lemma of the verb that is negated.
                        if another verb is encountered, the function is recursively called with the new current_verb
        - current_index: same as above
        Passing both current_verb and current_index to omit the search for the index's lemma in the tree.
        We need the index to be able to localize the tokens of the verb in RoBERTa tokenization and the lemma to be able
        to fill in verb_embeddings
        - in_clause: True if we are in a dependent clause
        - n: current recursion depth"""
        nonlocal phrase  # this is the linearized tree (phrase in the first for loop -
        # we have "for phrase in dependency_trees")
        # only needed for localizing "no more" and "no longer"
        if n >= rec_depth:
            print(f"Discarded: {phrase}\nExceeded recursion depth.")
            return 1

        #nonlocal representations  # will be initialized for each phrase; the RoBERTa encoding
        nonlocal negation_found  # will be initialized for each phrase; dictionary that tells us
        # whether negation was found for each verb lemma (for each auxilary)

        # the three following variables are initialized in collect_context_representations.py
        nonlocal num_complex_ph
        nonlocal num_neg
        nonlocal num_negations_in_dependent_cl

        nonlocal clause_found  # a bool, signals whether the phrase was already counted in num_complex_phrases; defined
        # in the loop within get_verb_embeddings

        if root.token['upos'] == "VERB":  # the function is called for all children but current_verb
            # and current_index change
            negation_found[root.token['id']] = [0, 0]  # a verb is found but the negation not yet -
            # the auxiliaries also haven't because we are in a tree structure
            if len(root.children) > 0:
                for child in root.children:
                    if root.token['deprel'].startswith(("ccomp", "xcomp", "csubj", "advcl", "acl", "list", "parataxis")):
                        if not clause_found:
                            num_complex_ph += 1
                            clause_found = True
                        depth_search(child, root.token['lemma'], root.token['id'], True, n + 1)
                    else:
                        depth_search(child, root.token['lemma'], root.token['id'], False, n + 1)

        else:  # we haven't found negation and root is not a verb
            # we iterate through the root's children, not changing the other parameters
            if root.token['upos'] == "AUX":  # we have found an auxiliary
                if current_index in negation_found:  # we check if it is really the auxiliary of current_verb
                    negation_found[current_index][0] += 1
                if root.token['lemma'] == 'cannot' or root.token['lemma'] == "can't":
                    if current_index in negation_found:
                        negation_found[current_index][1] += 1
                    else:
                        negation_found[current_index] = [1, 1]

            # we check whether we have found negation:
            elif root.token['lemma'] == 'not':
                '''   or root.token['lemma'] == 'never' or (
                    root.token['lemma'] == 'more' and root.token['id'] > 1 and phrase[root.token['id'] - 2]['lemma'] == 'no') or (
                    root.token['lemma'] == 'longer' and root.token['id'] > 1 and phrase[root.token['id'] - 2]['lemma'] == 'no'):'''

                if current_index in negation_found:  # it is possible for the head to be something other than a verb,
                    # for example in the phrase "Martin will have no more apple sauce"
                    # where the head of negation is "sauce" - in this case we will ignore it
                    negation_found[current_index][1] += 1  # a negation found for the current verb
                    num_neg += 1
                    if in_clause:
                        num_negations_in_dependent_cl += 1

            # iterate though all the children (there is probably no children here)
            if len(root.children) > 0:
                for child in root.children:
                    if root.token['deprel'].startswith(("ccomp", "xcomp", "csubj", "advcl", "acl", "list", "parataxis")):
                        if not clause_found:
                            num_complex_ph += 1
                            clause_found = True
                        depth_search(child, current_verb, current_index, True, n + 1)
                    else:
                        depth_search(child, current_verb, current_index, False, n + 1)

    # reading the file from path
    f = open(path)
    dep_trees = conllu.parse_incr(f)

    tot_pos = 0
    tot_neg = 0

    # verb_embeddings is a map: lemma of the verb v -> [context_reprs_negated, context_reprs_affirmative]

    # we have List[List[torch.Tensor]] since it is possible that some verbs be split into multiple tokens in RoBERTa

    num_ph, num_complex_ph, num_neg, num_negations_in_dependent_cl, disc = 0, 0, 0, 0, 0



    for phrase in dep_trees:
        print("LLLL")
        print(phrase)
        num_ph += 1
        if num_ph % 1000 == 0:
            print(f"{num_ph} at {datetime.now().strftime('%H:%M:%S')}, nb_neg : {tot_neg}, tot_pos : {tot_pos }")
            # print(f"Memory usage: {torch.cuda.memory_allocated(device)}")


        phrase_tree = phrase.to_tree()
        print("LLLL2")
        print(phrase_tree)

        if 'lemma' not in phrase_tree.token.keys() or 'id' not in phrase_tree.token.keys():
            continue

        # tokenizing and encoding of the original phrase using RoBERTa
        with torch.no_grad():
            bert_tokens = tokenizer(phrase_tree.metadata['text'], return_tensors='pt',
                                    max_length=512, padding=True, truncation=True).to(device)


        # getting the stanza to RoBERTa token map
        try:
            token_mapping = stanza_to_bert_tokens(phrase, tokenizer(phrase_tree.metadata['text'])['input_ids'],
                                                  tokenizer)
        except:
            print(f"Token mapping error on sentence: {phrase}")
            continue

        negation_found = {}  # Dict[int, [int, int]], maps the index of a verb to  a tuple (num_aux, num_negations) -
        # the number of auxiliaries and the number of negations of the verb)

        clause_found = False
        # depth first search from the tree: see function above

        try:
            lim = depth_search(phrase_tree, phrase_tree.token['lemma'], phrase_tree.token['id'], False)
            if lim is not None:
                print(f"Recursion depth exceeded at sentence {phrase}")
                continue
        except:
            print(f"Analysing error on phrase {phrase}")
            continue

        # current_verbs are now filled
        for index in negation_found:
            lemma = phrase[index - 1]['lemma']

            start, end = token_mapping[index - 1]  # localizing the verb in the RoBERTa tokenization

            # because the encodings are truncated to 512!
            if start >= 512 or end >= 512:
                disc += 1
                continue


            verb_to_add = verb_to_add.detach().cpu()


            if tot_neg >= nb_verbs and tot_pos >= nb_verbs:
                return list_sentences

            if negation_found[index][1] == 0:
                if random.random()<0.99:
                    continue




            if negation_found[index][1] == 0:  # negation wasn't found for the verb at position index
                tot_neg += 1
                if lemma not in verb_embs:
                    verb_embs[lemma] = [[], [verb_to_add]]
                else:
                    verb_embs[lemma][1].append(verb_to_add)
            elif negation_found[index][1] >= negation_found[index][0]:  # the number of negations is
                # bigger than or equal to the number of auxiliaries
                tot_pos += 1
                if lemma not in verb_embs:
                    verb_embs[lemma] = [[verb_to_add], []]
                else:
                    verb_embs[lemma][0].append(verb_to_add)
            else:  # then negations were found but not for every auxiliary, thus we add the tensors to both sides
                tot_neg += 1
                tot_pos += 1
                if lemma not in verb_embs:
                    verb_embs[lemma] = [[verb_to_add], [verb_to_add]]
                else:
                    verb_embs[lemma][0].append(verb_to_add)
                    verb_embs[lemma][1].append(verb_to_add)

    # we have exited the first loop, everything we need is in verb_embs
    return list_sentences


with torch.no_grad():


    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    abs_path = "/data/mnedeljkovic/thesis/thesis/code"

    nb_verbs = int(sys.argv[1])



    total_phrases, total_complex_phrases, total_negations, total_negations_in_dependent_clauses, total_discarded = 0, 0, 0, 0, 0

    first_page = 0


    while total_negations< nb_verbs :

        dependency_trees = f"{abs_path}/parsed/parsed{first_page}.conll"  # the file with parsed phrases

        num_phrases, num_complex_phrases, num_negations, num_negations_in_dependent_clauses, discarded =\
            get_sentences(dependency_trees, tokenizer)


        print(f"Total phrases: {total_phrases}")
        print(f"total_negations: {total_negations}")

        total_phrases += num_phrases
        total_complex_phrases += num_complex_phrases
        total_negations += num_negations
        total_negations_in_dependent_clauses += num_negations_in_dependent_clauses
        total_discarded += discarded

        first_page += 10000


    torch.save(embeddings, f"embeddings/embeddings_{first_page}_{nb_verbs}")




