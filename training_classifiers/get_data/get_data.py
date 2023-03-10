import torch



path  = ""


pages = 10000
embedds = torch.load(f"{path}/embeddings_{pages}")

print(type(embedds))