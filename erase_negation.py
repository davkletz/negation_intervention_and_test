import sys
sys.path.append("../src")
from src import debias
from joblib import load, dump
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression, PassiveAggressiveClassifier
from joblib import dump
import torch


model_name = "roberta-large"


def tsne(vecs, labels, title="", ind2label = None, words = None, metric = "l2"):

  tsne = TSNE(n_components=2)#, angle = 0.5, perplexity = 20)
  vecs_2d = tsne.fit_transform(vecs)
  #label_names = sorted(list(set(labels.tolist())))
  label_names = sorted(list(set(labels.tolist())))
  num_labels = len(label_names)

  names = sorted(set(labels.tolist()))

  plt.figure(figsize=(6, 5))
  colors = "red", "blue"
  for i, c, label in zip(sorted(set(labels.tolist())), colors, names):
     plt.scatter(vecs_2d[labels == i, 0], vecs_2d[labels == i, 1], c=c,
                label=label if ind2label is None else ind2label[label], alpha = 0.3, marker = "s" if i==0 else "o")
     plt.legend(loc = "upper right")

  plt.title(title)
  plt.savefig(f"embeddings_{title}.png", dpi=600)
  plt.show()
  return vecs_2d


path = "/data/dkletz/Experiences/negation_intervention_and_test/training_classifiers/get_data/embeddings"

vects = load(f"{path}/new_vectors_Rob")
labs = load(f"{path}/new_labs_Rob")



vects = np.array(vects)
labs = np.array(labs)
train_vects, test_vects, train_labs, test_labs = train_test_split(vects, labs)

print(train_vects.shape)
print(vects[0].shape)

print(test_vects.shape)
print(vects[0].shape)



M =  2000
ind2label =  {1: "NOT present", 0: "NOT absent"}
print("a")
print(train_labs[:M])
tsne_before = tsne(train_vects[:M], train_labs[:M], title = "Original (t=0)", ind2label =ind2label )
