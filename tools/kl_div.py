
import torch.nn.functional as F
import torch.nn as nn
import torch


def compute_kl_div(distr_orig, distrib_modif):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # input should be a distribution in the log space
    #to_comp = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
    # Sample a batch of distributions. Usually this would come from the dataset
    #target = F.softmax(torch.rand(3, 5), dim=1)
    output = kl_loss(distr_orig, distrib_modif)

    '''kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_target = F.log_softmax(torch.rand(3, 5), dim=1)
    output = kl_loss(to_comp, log_target)
    '''


    return output



