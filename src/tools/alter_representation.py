import numpy as np
import scipy
import torch
from time import time

def get_rowspace_projection(W: torch.Tensor) -> torch.Tensor:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """


    #current_W = W.cpu().numpy()
    current_W = W.cpu()
    if np.allclose(current_W, 0):
        w_basis = torch.zeros_like(current_W.T)
    else:
        w_basis = scipy.linalg.orth(current_W.T) # orthogonal basis

    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

    return P_W



def alter_representation(representations_to_alter, P, Ws : torch.Tensor, cudastring, direction = -1, alpha = 1):

    time_0 = time()

    neutral_rpz = (P.matmul(representations_to_alter.T)).T
    time_1 = time()
    #print(f"temps intermédiaire 1 : {time_1 - time_0} ")
    feature_rpz = torch.zeros(neutral_rpz.shape).to(cudastring)

    #print(f"shape WS : {len(Ws)}")

    time_2 = time()


    for w_vect in Ws:

        scal = (w_vect.matmul(representations_to_alter.T)).T

        coeff = -torch.ones(scal.shape).to(cudastring)


        coeff[torch.where(torch.sign(scal) == torch.sign(torch.tensor(direction)))] = 1

        w_proj_mat = torch.tensor(get_rowspace_projection(w_vect)).to(cudastring)
        prof_feature = (w_proj_mat.matmul(representations_to_alter.T)).T
        feature_rpz += coeff*prof_feature

    time_3 = time()

    #print(f"temps intermédiaire 3 : {time_3 - time_2} ")

    counter_repz = neutral_rpz + alpha*feature_rpz

    return counter_repz


