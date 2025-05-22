import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLossCosine(nn.Module):
    def __init__(self, margin=0.5):

        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):

        batch_size = embeddings.size(0)
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)  # (batch_size, batch_size)

        labels = labels.unsqueeze(1)  # (batch_size, 1)
        label_matrix = (labels != labels.T).float()  # (batch_size, batch_size)

        positive_loss = (1 - label_matrix) * (1 - cosine_sim)
        negative_loss = label_matrix * F.relu(cosine_sim - self.margin)

        loss = (positive_loss + negative_loss).sum() / (batch_size * (batch_size - 1)) 
        
        return loss