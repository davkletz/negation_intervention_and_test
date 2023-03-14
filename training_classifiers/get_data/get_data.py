import torch
import numpy as np


first_page = 10000
#abs_path = "/data/mnedeljkovic/thesis/thesis/code"
emb_path = f"embeddings/embeddings{first_page}_10000"

pages = 10000


dico_detail = {}

tot_p = 0
tot_n = 0


print(dico_detail)

print(tot_p, tot_n)


def get_data():
    """
    :return: X_train, Y_train, X_dev, Y_dev
    """
    embedds: dict = torch.load(f"{emb_path}")

    list_pos = []
    list_neg = []



    for key in embedds.keys():
        list_pos.extend(embedds[key][0])
        list_neg.extend(embedds[key][1])

    array_pos = np.stack(list_pos)
    array_neg = np.stack(list_neg)

    return array_neg, array_pos


a, b = get_data()