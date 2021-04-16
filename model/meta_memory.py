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

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

def sum_distance(a, b):
    return ((a - b) ** 2).sum(-1)

def mean_distance(a, b, weight=None, training=True):
    dis = ((a - b) ** 2).sum(-1)

    if weight is not None:
        dis *= weight 

    if not training:
        return dis
    else:
        return dis.mean().unsqueeze(0)

def max_distance(a, b):
    return ((a - b) ** 2).sum(-1).max().unsqueeze(0)

def distance(a, b):
    return ((a - b) ** 2).sum(-1)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))

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

class Meta_Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather, shrink_thres=0):
        super(Meta_Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        #multi-head
        self.Mheads = nn.Linear(key_dim, memory_size, bias=False)
        # self.Dim_reduction = nn.Linear(key_dim, feature_dim)
        # self.softmax = nn.Softmax(dim=1)
        self.shrink_thres = shrink_thres
        
    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem,torch.t(self.keys_var))
        similarity[:,i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)
        
        
        return self.keys_var[max_idx]
    
    def get_score(self, mem, query):
        bs, n, d = query.size()#n=w*h
        bs, m, d = mem.size()
        # import pdb;pdb.set_trace()
        score = torch.bmm(query, mem.permute(0,2,1))# b X h X w X m
        score = score.view(bs, n, m)# b X n X m
        
        score_query = F.softmax(score, dim=1)
        score_memory = F.softmax(score, dim=2)
        
        return score_query, score_memory
    
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
                multi_heads_weights = linear(key, weights['memory.Mheads.weight'])

            # multi_heads_weights = self.Mheads(key)
            multi_heads_weights = multi_heads_weights.view((batch_size, h*w, self.memory_size, 1))

            # softmax on weights
            multi_heads_weights = F.softmax(multi_heads_weights,dim=1)
            # hard_shrink
            if self.shrink_thres>0:
                multi_heads_weights = hard_shrink_relu(multi_heads_weights, lambd=self.shrink_thres)
                # normalize
                multi_heads_weights = F.normalize(multi_heads_weights, p=1, dim=1)

            # import pdb;pdb.set_trace()
            key = key.reshape((batch_size,w*h,dims))
            mems = multi_heads_weights*key.unsqueeze(-2)
            mems = mems.sum(1)
            
            # mems = F.normalize(mems, dim=-1)
            # import pdb;pdb.set_trace()
            # losses
            updated_query, fea_loss, cst_loss, dis_loss = self.query_loss(query, mems, weights, train)
            # import pdb;pdb.set_trace()
            # skip connection
            updated_query = updated_query+query

            # reshape
            updated_query = updated_query.permute(0,2,1) # b X d X n
            updated_query = updated_query.view((batch_size, self.feature_dim, h_, w_))
            return updated_query, mems, fea_loss, cst_loss, dis_loss
        
        #test
        else:
            if weights == None:
                multi_heads_weights = self.Mheads(key)
            else:
                multi_heads_weights = linear(key, weights['memory.Mheads.weight'])
                
            multi_heads_weights = multi_heads_weights.view((batch_size, h*w, self.memory_size, 1))

            # softmax on weights
            multi_heads_weights = F.softmax(multi_heads_weights,dim=1)
            
            # import pdb;pdb.set_trace()
            # hard_shrink
            if self.shrink_thres>0:
                multi_heads_weights = hard_shrink_relu(multi_heads_weights, lambd=self.shrink_thres)
                # normalize
                multi_heads_weights = F.normalize(multi_heads_weights, p=1, dim=1)

            # heatmap(multi_heads_weights)
            # import pdb;pdb.set_trace()
            key = key.reshape((batch_size,w*h,dims))
            mems = multi_heads_weights*key.unsqueeze(-2)
            mems = mems.sum(1)
            # mems = self.Dim_reduction(mems)
            # heatmap(norm.unsqueeze(-1))
            
            # normlize
            # query = F.normalize(query, dim=-1)
            # mems = F.normalize(mems, dim=-1)

            # loss
            updated_query, fea_loss, query, softmax_score_query, softmax_score_memory = self.query_loss(query, mems, weights, train)
            # import pdb;pdb.set_trace()
            # skip connection
            updated_query = updated_query+query
            # reshape
            updated_query = updated_query.permute(0,2,1) # b X d X n
            updated_query = updated_query.view((batch_size, self.feature_dim, h_, w_))
            return updated_query, mems, softmax_score_query, softmax_score_memory, query, fea_loss
        
    def query_loss(self, query, keys, weights, train):
        batch_size, n, dims = query.size() # b X n X d, n=w*h
        if train:
            
            # Distinction constrain
            keys_ = F.normalize(keys, dim=-1)
            similarity = torch.bmm(keys_, keys_.permute(0,2,1))
            dis = 1-distance(keys_.unsqueeze(1), keys_.unsqueeze(2))
            
            mask = dis>0
            dis *= mask.float()
            dis = torch.triu(dis, diagonal=1)
            dis_loss = dis.sum(1).sum(1)*2/(self.memory_size*(self.memory_size-1))
            dis_loss = dis_loss.mean()

            # maintain the consistance of same attribute vector
            cst_loss = mean_distance(keys_[1:], keys_[:-1])

            # Normal constrain
            loss_mse = torch.nn.MSELoss()
            # Dim reduction
            # if weights == None:
            #     keys = self.Dim_reduction(keys)
            # else:
            #     keys = linear(keys, weights['memory.Dim_reduction.weight'], weights['memory.Dim_reduction.bias'])
            keys = F.normalize(keys, dim=-1)
            _, softmax_score_memory = self.get_score(keys, query)
            # import pdb;pdb.set_trace()
            # hard_shrink
            # if self.shrink_thres>0:
            #     att_weight = hard_shrink_relu(softmax_score_memory, lambd=self.shrink_thres)
            #     # normalize
            #     att_weight = F.normalize(att_weight, p=1, dim=-1)
            # else:
            #      att_weight = softmax_score_memory

            new_query = softmax_score_memory.unsqueeze(-1)*keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)
            # reconstrcut feature vectors
            # fea_loss = loss_mse(query, new_query)
            # fea_loss = mean_distance(query, new_query, None, train)
            # import pdb;pdb.set_trace()

            

            # import pdb;pdb.set_trace()
            # maintain the distinction among attribute vectors
            _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=-1)
        
            # 1st, 2nd closest memories
            pos = torch.gather(keys,1,gathering_indices[:,:,:1].repeat((1,1,dims)))
            # neg = torch.gather(keys,1,gathering_indices[:,:,1:2].repeat((1,1,dims)))
            # dis_loss = loss_dis(query, pos, neg)
            # import pdb;pdb.set_trace()
            fea_loss = loss_mse(query, pos)

            
            # import pdb;pdb.set_trace()
            return new_query, fea_loss, cst_loss, dis_loss
        
            
        else:
            loss_mse = torch.nn.MSELoss(reduction='none')
            # if weights == None:
            #     keys = self.Dim_reduction(keys)
            # else:
            #     keys = linear(keys, weights['memory.Dim_reduction.weight'], weights['memory.Dim_reduction.bias'])
            keys = F.normalize(keys, dim=-1)
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
                
            # hard_shrink
            # if self.shrink_thres>0:
            #     att_weight = hard_shrink_relu(softmax_score_memory, lambd=self.shrink_thres)
            #     # normalize
            #     att_weight = F.normalize(att_weight, p=1, dim=-1)
            # else:
            #     att_weight = softmax_score_memory

            new_query = softmax_score_memory.unsqueeze(-1)*keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)
            # import pdb;pdb.set_trace()
            # fea_loss = loss_mse(query, new_query)
            # fea_loss = mean_distance(query, new_query, None, train)
            # n_query = query.mean(1,keepdim=True)
            # fea_loss = sum_distance(query, n_query)
            # fea_loss = torch.var(fea_loss,-1)
            # import pdb;pdb.set_trace()
            # query_repeat = query.unsqueeze(-2).repeat((1,1,keys.shape[1],1))
            # keys_repeat = keys.unsqueeze(1).repeat((1,query.shape[1],1,1))
            
            # import pdb;pdb.set_trace()
            # pq = torch.bmm(query, new_query.permute(0,2,1))
            _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=-1)
        
            # # 1st, 2nd closest memories
            pos = torch.gather(keys,1,gathering_indices[:,:,:1].repeat((1,1,dims)))
            # neg = torch.gather(keys,1,gathering_indices[:,:,1:2].repeat((1,1,dims)))
            # import pdb;pdb.set_trace()
            # fea_loss = mean_distance(query, pos, training=train)
            fea_loss = loss_mse(query, pos)
            # fea_loss = torch.topk(fea_loss, 4, dim=-1, largest=False)[0].mean()
            # dis_loss = loss_dis(query, pos, neg)
            # similarity = torch.bmm(keys, keys.permute(0,2,1))
            # dis = 1-distance(keys.unsqueeze(1), keys.unsqueeze(2))
            
            # mask = dis>0
            # dis *= mask.float()
            # dis = torch.triu(dis, diagonal=1)
            # dis_loss = dis.sum(1).sum(1)*2/(self.memory_size*(self.memory_size-1))
            # dis_loss = dis_loss.mean()
            # import pdb;pdb.set_trace()
            return new_query, fea_loss, query, softmax_score_query, softmax_score_memory
    
    
