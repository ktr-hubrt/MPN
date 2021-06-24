import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import functional as F
from .layers import *


def mean_distance(a, b, weight=None, training=True):
    dis = ((a - b) ** 2).sum(-1)

    if weight is not None:
        dis *= weight 

    if not training:
        return dis
    else:
        return dis.mean().unsqueeze(0)

def distance(a, b):
    return ((a - b) ** 2).sum(-1)
    
def heatmap(x, name='heatmap'):
    # import pdb;pdb.set_trace()
    x = x.squeeze(-1)
    for j in range(x.shape[2]):
        plt.cla()
        y = x[0,:,j].reshape((32,32))
        df = pd.DataFrame(y.data.cpu().numpy())
        sns.heatmap(df)
        plt.savefig('results/heatmap/{}_{}.png'.format(name,str(j)))
        plt.close()
    return True

class Meta_Prototype(nn.Module):
    def __init__(self, proto_size, feature_dim, key_dim, temp_update, temp_gather, shrink_thres=0):
        super(Meta_Prototype, self).__init__()
        # Constants
        self.proto_size = proto_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        #multi-head
        self.Mheads = nn.Linear(key_dim, proto_size, bias=False)
        # self.Dim_reduction = nn.Linear(key_dim, feature_dim)
        # self.softmax = nn.Softmax(dim=1)
        self.shrink_thres = shrink_thres
    
    def get_score(self, pro, query):
        bs, n, d = query.size()#n=w*h
        bs, m, d = pro.size()
        # import pdb;pdb.set_trace()
        score = torch.bmm(query, pro.permute(0,2,1))# b X h X w X m
        score = score.view(bs, n, m)# b X n X m
        
        score_query = F.softmax(score, dim=1)
        score_proto = F.softmax(score, dim=2)
        
        return score_query, score_proto
    
    def forward(self, key, query, weights, train=True):

        batch_size, dims, h, w = key.size() # b X d X h X w
        key = key.permute(0,2,3,1) # b X h X w X d
        _, _, h_, w_ = query.size()
        query = query.permute(0,2,3,1) # b X h X w X d
        query = query.reshape((batch_size,-1,self.feature_dim))
        #train
        if train:
            if weights == None:
                multi_heads_weights = self.Mheads(key)
            else:
                multi_heads_weights = linear(key, weights['prototype.Mheads.weight'])


            multi_heads_weights = multi_heads_weights.view((batch_size, h*w, self.proto_size, 1))

            # softmax on weights
            multi_heads_weights = F.softmax(multi_heads_weights,dim=1)

            key = key.reshape((batch_size,w*h,dims))
            protos = multi_heads_weights*key.unsqueeze(-2)
            protos = protos.sum(1)
            
            updated_query, fea_loss, cst_loss, dis_loss = self.query_loss(query, protos, weights, train)

            # skip connection
            updated_query = updated_query+query

            # reshape
            updated_query = updated_query.permute(0,2,1) # b X d X n
            updated_query = updated_query.view((batch_size, self.feature_dim, h_, w_))
            return updated_query, protos, fea_loss, cst_loss, dis_loss
        
        #test
        else:
            if weights == None:
                multi_heads_weights = self.Mheads(key)
            else:
                multi_heads_weights = linear(key, weights['prototype.Mheads.weight'])
                
            multi_heads_weights = multi_heads_weights.view((batch_size, h*w, self.proto_size, 1))

            # softmax on weights
            multi_heads_weights = F.softmax(multi_heads_weights,dim=1)

            key = key.reshape((batch_size,w*h,dims))
            protos = multi_heads_weights*key.unsqueeze(-2)
            protos = protos.sum(1)

            # loss
            updated_query, fea_loss, query = self.query_loss(query, protos, weights, train)

            # skip connection
            updated_query = updated_query+query
            # reshape
            updated_query = updated_query.permute(0,2,1) # b X d X n
            updated_query = updated_query.view((batch_size, self.feature_dim, h_, w_))
            return updated_query, protos, query, fea_loss
        
    def query_loss(self, query, keys, weights, train):
        batch_size, n, dims = query.size() # b X n X d, n=w*h
        if train:
            
            # Distinction constrain
            keys_ = F.normalize(keys, dim=-1)
            dis = 1-distance(keys_.unsqueeze(1), keys_.unsqueeze(2))
            
            mask = dis>0
            dis *= mask.float()
            dis = torch.triu(dis, diagonal=1)
            dis_loss = dis.sum(1).sum(1)*2/(self.proto_size*(self.proto_size-1))
            dis_loss = dis_loss.mean()

            # maintain the consistance of same attribute vector
            cst_loss = mean_distance(keys_[1:], keys_[:-1])

            # Normal constrain
            loss_mse = torch.nn.MSELoss()

            keys = F.normalize(keys, dim=-1)
            _, softmax_score_proto = self.get_score(keys, query)

            new_query = softmax_score_proto.unsqueeze(-1)*keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)

            # maintain the distinction among attribute vectors
            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)
        
            # 1st closest memories
            pos = torch.gather(keys,1,gathering_indices[:,:,:1].repeat((1,1,dims)))

            fea_loss = loss_mse(query, pos)

            return new_query, fea_loss, cst_loss, dis_loss
        
            
        else:
            loss_mse = torch.nn.MSELoss(reduction='none')

            keys = F.normalize(keys, dim=-1)
            softmax_score_query, softmax_score_proto = self.get_score(keys, query)

            new_query = softmax_score_proto.unsqueeze(-1)*keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)

            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)
        
            #1st closest memories
            pos = torch.gather(keys,1,gathering_indices[:,:,:1].repeat((1,1,dims)))

            fea_loss = loss_mse(query, pos)

            return new_query, fea_loss, query
    
    
