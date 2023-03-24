import torch
import numpy as np


first_page = 10000
#abs_path = "/data/mnedeljkovic/thesis/thesis/code"
emb_path = f"get_data/embeddings/embeddings_{first_page}_10000"
#
emb_path=f"/data/dkletz/Experiences/negation_intervention_and_test/training_classifiers/get_data/embeddings/embeddings_{first_page}_10000"

pages = 10000


dico_detail = {}

tot_p = 0
tot_n = 0





def get_data():
    """
    :return: X_train, Y_train, X_dev, Y_dev
    """
    embedds: dict = torch.load(f"{emb_path}")

    list_pos = []
    list_neg = []


    for key in embedds.keys():
        list_pos.extend(embedds[key][1])
        list_neg.extend(embedds[key][0])

    array_pos = np.stack(list_pos)
    array_neg = np.stack(list_neg)


    size = min(len(array_pos), len(array_neg))

    array_pos = array_pos[:size]
    array_neg = array_neg[:size]

    labs = [0 for k in range(len(array_neg))]
    labs.extend([1 for k in range(len(array_pos))])
    arrays = np.concatenate((array_neg, array_pos), axis=0)

    labs_np = np.array(labs)

    return arrays, labs_np
