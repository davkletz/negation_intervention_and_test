import torch



path  = "/data/mnedeljkovic/thesis/thesis/code/embeddings"


pages = 10000
embedds = torch.load(f"{path}/embeddings_{pages}")

print(type(embedds))