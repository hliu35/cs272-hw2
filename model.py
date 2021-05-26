import numpy as np
import torch



def build_model(ratings, embedding_dim = 4):
    pass


def custom_loss(A, A_pred):
    loss_all = torch.square(A-A_pred)
    return torch.sum(loss_all)


class MFNet(torch.nn.Module):
    
    def __init__(self, user_count, item_count, embedding_dim):
        super().__init__()
        self.U = torch.nn.Embedding(user_count, embedding_dim)
        self.V = torch.nn.Embedding(item_count, embedding_dim)
        self.A_pred = torch.nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, user_idx, item_idx):
        user_embedding = self.U(user_idx)
        item_embedding = self.V(item_idx)
        element_wise_product = torch.mul(user_embedding, item_embedding)
        prediction = self.A_pred(element_wise_product)

        return prediction
