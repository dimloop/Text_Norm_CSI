import torch
import torch.nn as nn

class SimilarityLoss(nn.Module):
    """
    Similarity loss function for Siamese networks.
    Encourages similar pairs to have small distances and
    dissimilar pairs to have distances larger than a margin.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8  # Small value to avoid log(0)

    def forward(self, p, distance, y):


        similar_loss = (1 - y) * torch.log(p + self.eps)  # add small value to avoid log(0)
       
        dissimilar_loss = y * torch.log(1- p + self.eps)  # add small value to avoid log(0)
  

        loss = torch.mean(similar_loss + dissimilar_loss) 

        return -loss
    

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese networks.
    Encourages similar pairs to have small distances and
    dissimilar pairs to have distances larger than a margin.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, p, distance, y):
       

        # loss for similar pairs: distance^2
        similar_loss = (1-y) * torch.pow(distance, 2)

        # loss for dissimilar pairs: max(0, margin - distance)^2
        dissimilar_loss =  y * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        # mean loss across the batch
        loss = torch.mean(similar_loss + dissimilar_loss) 

        return loss
    
