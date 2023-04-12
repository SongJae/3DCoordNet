# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, GENConv, DeepGCNLayer, XConv
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from molx.dataset.utils import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims() #여기서 10은 random_vec_dim을 의미
full_bond_feature_dims = get_bond_feature_dims()

def make_mask(batch, device):
    n = batch.shape[0]
    mask = torch.eq(batch.unsqueeze(1), batch.unsqueeze(0))
    mask = (torch.ones((n, n)) - torch.eye(n)).to(device) * mask
    count = torch.sum(mask)
    return mask, count

def make_angle_mask(batch, num_batches, device):
  n = batch.shape[0] # n은 하나의 batch에 있는 전체 원자의 개수를 의미
  mask = torch.zeros([n, n-1, n-1], dtype = torch.float32, device = device)
  
  output, counts = torch.unique(batch, sorted = True, return_counts = True)
  
  atom_index = 0
  for j in range(len(counts)):
    mask[atom_index:atom_index+counts[j], atom_index:atom_index + counts[j] - 1, atom_index:atom_index + counts[j] - 1] = 1
    atom_index = atom_index + counts[j]
  for j in range(len(mask)):
    mask[j].fill_diagonal_(0)
  
  count = torch.sum(mask)
  return mask, count

class Deepergcn_dagnn_dist(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, 
                 JK="last", aggr='softmax', norm='batch', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Deepergcn_dagnn_dist, self).__init__()

        self.deepergcn_dagnn = DeeperDAGNN_node_Virtualnode(num_layers, emb_dim, drop_ratio, JK, aggr, norm)
        self.calc_dist = DistMax(emb_dim)

    def forward(self, batched_data, train=False):
        xs = self.deepergcn_dagnn(batched_data)
        mask_d_pred, mask, count = self.calc_dist(xs, batched_data.batch)
        return mask_d_pred, mask, count


class DistMax(torch.nn.Module):
    def __init__(self, emb_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DistMax, self).__init__()
        self.fc = torch.nn.Linear(in_features=emb_dim, out_features=1)
        self.device = device

    def forward(self, xs, batch, train=False):
        d_pred = self.fc(torch.max(xs.unsqueeze(0), xs.unsqueeze(1))).squeeze()
        mask, count = make_mask(batch, self.device)

        if train:
            mask_d_pred = d_pred * mask
        else:
            mask_d_pred = F.relu(d_pred * mask)
        return mask_d_pred, mask, count


class Deepergcn_dagnn_coordinate(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5,
                 JK="last", aggr='softmax', norm='batch',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Deepergcn_dagnn_coordinate, self).__init__()

        self.deepergcn_dagnn = DeeperDAGNN_node_Virtualnode(num_layers, emb_dim, drop_ratio, JK, aggr, norm)
        self.calc_dist_angle = DistAngleCoords(emb_dim)

    def forward(self, batched_data, batch_size, train=False):
        xs = self.deepergcn_dagnn(batched_data)
        xs = self.calc_dist_angle(xs, batched_data.batch, batch_size)
        return xs


class DistAngleCoords(torch.nn.Module):
    def __init__(self, emb_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DistAngleCoords, self).__init__()
        self.fc = torch.nn.Linear(in_features=emb_dim, out_features=3)
        self.device = device

    def forward(self, xs, batch, batch_size, train=False):
        #num_atoms는 batch별로 분자를 이루는 원자의 개수 나타냄
        
        xs = self.fc(xs)



        return xs


class DAGNN(MessagePassing):
    def __init__(self, K, emb_dim, normalize=True, add_self_loops=True):
        super(DAGNN, self).__init__()
        self.K = K
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.proj = torch.nn.Linear(emb_dim, 1)

        self._cached_edge_index = None

    def forward(self, x, edge_index, edge_weight=None):
        if self.normalize:
            edge_index, norm = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
            

        return bond_embedding   


class DeeperDAGNN_node_Virtualnode(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", aggr='softmax', norm='batch', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(DeeperDAGNN_node_Virtualnode, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # self.input_encode_manner = input_encode_manner

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        self.dagnn = DAGNN(5, emb_dim)

        ###List of GNNs
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            #if i==1:
            #    conv = XConv(emb_dim, emb_dim, dim=2, kernel_size=1)
            #else:
            if i==1:
                conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=True, num_layers=5, norm=norm)
            else:    
                conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=True, num_layers=5, norm=norm)


            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)


            layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.layers.append(layer)

        for layer in range(num_layers - 1):
            if norm=="batch":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                                     torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
            elif norm=="layer":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU(), \
                                                                     torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU()))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch, pos = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch, batched_data.pos
        
        print("first input x: ", x, x.shape)
        print("first input edge_attr: ", edge_attr, edge_attr.shape)
        print("batch: ", batch)
       
        #여기서부터 이제 random noise 첨가하는 과정으로
        
        """
        rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
        
        rand_x = rand_dist.sample([10, x.size(0), 10]).squeeze(-1).to(self.device) # added squeeze
        rand_edge = rand_dist.sample([10, edge_attr.size(0), 3]).squeeze(-1).to(self.device) # added squeeze
        x = torch.cat([x.unsqueeze(0).repeat(10, 1, 1), rand_x], dim=-1)
        edge_attr = torch.add(edge_attr.unsqueeze(0).repeat(10, 1, 1), rand_edge)
        
        x = torch.reshape(x,(x.shape[0]*x.shape[1],-1))
        edge_attr = torch.reshape(edge_attr,(edge_attr.shape[0]*edge_attr.shape[1],-1))
        print("x before encoder: ", x, x.shape)    
        print("edge_attr before encoder: ", edge_attr, edge_attr.shape)
        """
        
        edge_attr = self.bond_encoder(edge_attr)
        h = self.atom_encoder(x)

        h_list = []

        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h = h + virtualnode_embedding[batch]

        #h = self.layers[0].conv(h, pos, batch)
        
        rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
        
        rand_h = rand_dist.sample([h.size(0), self.emb_dim]).squeeze(-1).to(self.device) # added squeeze
        rand_edge = rand_dist.sample([edge_attr.size(0), self.emb_dim]).squeeze(-1).to(self.device) # added squeeze
        #edge_index = edge_index.unsqueeze(1).repeat(1, 10, 1)
        #h = torch.cat([h, rand_h], dim=-1)
        #edge_attr = torch.cat([edge_attr, rand_edge], dim=-1)
        h = torch.add(h, rand_h)
        edge_attr = torch.add(edge_attr, rand_edge)
        
        """
        h = torch.reshape(h, (h.shape[0]*h.shape[1], -1))
        edge_attr = torch.reshape(edge_attr, (edge_attr.shape[0]*edge_attr.shape[1], -1))
        print("edge_index shape here: ", edge_index.shape)
        edge_index = torch.reshape(edge_index, (-1, edge_index.shape[1]*edge_index.shape[2]))
        print("h before convolution: ", h, h.shape)    
        print("edge_attr before convolution: ", edge_attr, edge_attr.shape)       
        print("edge_index before convolution: ", edge_index, edge_index.shape)
        """
        
        h = self.layers[0].conv(h, edge_index, edge_attr)

        h_list.append(h)
       
        
        for i, layer in enumerate(self.layers[1:]):
            print("h size: ", h.shape)
            print("edge_attr size: ", edge_attr.shape)
            h = layer(h, edge_index, edge_attr)

            ### update the virtual nodes
            ### add message from graph nodes to virtual nodes
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding

            ### transform virtual nodes using MLP
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

            h = h + virtualnode_embedding[batch]
            h_list.append(h)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0, training=self.training)

        h_list.append(h)
        h = h + virtualnode_embedding[batch]

        h = self.dagnn(h, edge_index)
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation

