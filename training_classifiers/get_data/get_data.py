import torch



path  = "/data/mnedeljkovic/thesis/thesis/code/embeddings"


pages = 10000
embedds : dict = torch.load(f"{path}/embeddings{pages}")

print(type(embedds))
print(embedds.keys())


dico_detail = {}

tot_p = 0
tot_n = 0

for key in embedds.keys():
    nb_p = len(embedds[key][1])
    nb_n = len(embedds[key][0])

    dico_detail[key] = (nb_p, nb_n)
    tot_n += nb_n
    tot_p += nb_p

print(dico_detail)

print(tot_p, tot_n)