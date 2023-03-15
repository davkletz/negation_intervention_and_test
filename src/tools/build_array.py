
def build_array(sourcefile):
    """"
    The function builds an array out of an inputfile,
    one array entry per line
    """
    array_in_construction=[]
    for line in sourcefile:
        #print(line)
        cleanline=line.strip('\n ')
        array_in_construction.append(cleanline)
    return array_in_construction



def build_hypo(sourcefile):
    """"
    The function builds sets of minimal positive-negative sentences
    """
    array_in_construction=[]
    nb_line = 0
    current_set = ()
    for line in sourcefile:
        cleanline=line.strip('\n')
        #array_in_construction.append(cleanline.split('\t'))
        current_set+=(cleanline, )

        if nb_line%4 == 3:
            array_in_construction.append(current_set)
            current_set = ()

        nb_line += 1
    return array_in_construction



"""filepath = "../../gubelman_FR"

thehnamefile=open(filepath,"r")
a = build_hypo(thehnamefile)
print(a)
"""


def build_masked_sentences(hypo_sentence_available, name_available, profession_available, top_token, current_pronouns_maj, current_pronoun, mask_token):
    new_hypo_sentence_available = hypo_sentence_available.replace('NOM', name_available)
    new_hypo_sentence_available = new_hypo_sentence_available.replace('MET', profession_available)
    new_hypo_sentence_available = new_hypo_sentence_available.replace('ACT', top_token)
    new_hypo_sentence_available = new_hypo_sentence_available.replace('PRON_maj', current_pronouns_maj)
    new_hypo_sentence_available = new_hypo_sentence_available.replace('PRON', current_pronoun)
    new_hypo_sentence_available = new_hypo_sentence_available.replace('MASK', mask_token)

    return new_hypo_sentence_available



def build_masked_context(name_available, profession_available, verb, current_pronouns_maj, mask_token):
    context_available = "NOM is MET who likes to ACT. PRON_maj is happy to MASK."

    new_context_sentence_available = context_available.replace('NOM', name_available)
    new_context_sentence_available = new_context_sentence_available.replace('MET', profession_available)
    new_context_sentence_available = new_context_sentence_available.replace('ACT', verb)
    new_context_sentence_available = new_context_sentence_available.replace('PRON_maj', current_pronouns_maj)
    new_context_sentence_available = new_context_sentence_available.replace('MASK', mask_token)




    return new_context_sentence_available