import numpy as np
# from sympy import N   
import torch
import torch.nn.functional as F

def l2_regularization(model):
    l2_loss = []
    for name, parameters in model.named_parameters(): 
        l2_loss.append((parameters ** 2).sum() / 2.0)
    # for module in model.modules():
    #         l2_loss.append((module.weight ** 2).sum() / 2.0)
    return sum(l2_loss)

class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = self.mse = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, embedding_1, embedding_2):
        return self.mse(embedding_1, embedding_2)

class InfoNCE(torch.nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.t = temperature

    def forward(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.t)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.t).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

# class InfoNCE_wo_neg(torch.nn.Module):
#     def __init__(self):
#         super(InfoNCE_wo_neg, self).__init__()

#     def forward(self, view1, view2):
#         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
#         pos_score = (view1 * view2).sum(dim=-1)
#         # pos_score = torch.exp(pos_score / self.t)
#         ttl_score = torch.matmul(view1, view2.transpose(0, 1))
#         ttl_score = torch.exp(ttl_score / self.t).sum(dim=1)
#         cl_loss = -torch.log(pos_score / ttl_score)
#         return torch.mean(cl_loss)

class BPR(torch.nn.Module):
    def __init__(self):
        super(BPR, self).__init__()

    def forward(self, user_embedding, item_embedding, u, i, j, user_embedding_ego=None, item_embedding_ego=None):
        user = user_embedding[u]
        pos_item = item_embedding[i]
        neg_item = item_embedding[j]
        assert torch.isnan(user).sum() == 0
        assert torch.isnan(pos_item).sum() == 0
        assert torch.isnan(neg_item).sum() == 0
        # -------- BPR loss

        # prediction_i = (user * pos_item).sum(dim=-1)
        # prediction_j = (user * neg_item).sum(dim=-1)
        # assert torch.isnan(prediction_i).sum() == 0
        # assert torch.isnan(prediction_j).sum() == 0

        # bpr_loss = -((prediction_i - prediction_j).sigmoid().log().mean())
        pos_scores = torch.mul(user, pos_item)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user, neg_item)
        neg_scores = torch.sum(neg_scores, dim=1)
        bpr_loss = torch.sum(torch.nn.functional.softplus(neg_scores - pos_scores))

        if user_embedding_ego==None or item_embedding_ego==None:
            reg_loss = ((user ** 2).sum(dim=-1) + (pos_item ** 2 + neg_item ** 2).sum(dim=-1)).mean()
        else:
            userEmb0 = user_embedding_ego[u]
            posEmb0 = item_embedding_ego[i]
            negEmb0 = item_embedding_ego[j]
            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(u))

        return bpr_loss, reg_loss

class Normalize(torch.nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power) + 1e-8
        norm = norm.sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class CCA_SSG(torch.nn.Module):
    def __init__(self, env, dataset):
        super().__init__()
        self.env = env
        self.warm_n_user = dataset.warm_n_user
        self.warm_m_item = dataset.warm_m_item

    def forward(self, emb_1, emb_2):
        emb_1 = (emb_1 - emb_1.mean(0)) / emb_1.std(0)
        emb_2 = (emb_2 - emb_2.mean(0)) / emb_2.std(0)

        c = torch.mm(emb_1.T, emb_2.detach())
        c1 = torch.mm(emb_1.T, emb_1)
        c2 = torch.mm(emb_2.T, emb_2)

        c = c / (self.warm_n_user + self.warm_m_item)
        c1 = c1 / (self.warm_n_user + self.warm_m_item)
        c2 = c2 / (self.warm_n_user + self.warm_m_item)

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(self.env.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        return loss_inv, loss_dec1, loss_dec2

class Noise_path_length_radio():
    def __init__(self, start_emb, env):
        self.start_emb = start_emb
        self.end_emb = start_emb
        self.previous_emb = start_emb
        self.current_emb = start_emb

        self.all_distance = torch.zeros(self.start_emb.shape[0]).to(env.device)
        self.short_distance = torch.zeros(self.start_emb.shape[0]).to(env.device)
        self.distance = torch.nn.PairwiseDistance(2)

        self.emb_count = 0

    def add_emb(self, emb):
        self.emb_count += 1
        self.previous_emb = self.current_emb
        self.current_emb = emb
        self.end_emb = self.current_emb
        self.all_distance += self.distance(self.current_emb, self.previous_emb)

        
    def get_final_distance(self):
        self.short_distance = self.distance(self.end_emb, self.start_emb)
        return self.short_distance

    def get_radio(self):
        self.get_final_distance()
        return (self.all_distance/self.short_distance).mean()

        
def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()