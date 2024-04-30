import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GATConv, LayerNorm, GCNConv, GATv2Conv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_max_pool
from torch.nn import Linear
from utils import debug_log

class GAT_skip_forward(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_hidden_layers, n_gatconvs, use_bn,k=128):

        super(GAT_skip_forward, self).__init__()
        self.node_feature_dim = nfeat
        self.hidden_dim = nhid
        self.output_dim = nclass
        self.drop = dropout
        self.heads = nheads
        self.k=k
        self.batch_size=256

        self.conv0 = GATv2Conv(self.node_feature_dim, self.hidden_dim, heads=nheads, dropout=self.drop)
        self.convs = nn.ModuleList()
        
        
        for i in range(n_gatconvs):
            self.convs.append(GATv2Conv(nheads * self.hidden_dim, self.hidden_dim, heads=nheads, concat=True))

        
        self.norm0 = nn.BatchNorm1d(self.node_feature_dim)
        self.norm1 = nn.BatchNorm1d(nheads * self.hidden_dim)
        self.norm2 = nn.BatchNorm1d(nheads * self.hidden_dim)
        self.norm3 = nn.BatchNorm1d(self.hidden_dim)
        
        self.lin0 = Linear(nheads*self.hidden_dim, self.hidden_dim)
        self.lin1 = Linear(self.hidden_dim, self.output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            print('reset_parameters', p)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index):
        x = self.norm0(x)
        x = self.conv0(x, edge_index)
        x = self.norm1(x)

        for i in range(len(self.convs) - 1):
            z = x
            print('1x.shape', x.shape)
            print('1z.shape', z.shape)
            x = self.convs[i](x, edge_index)
            print('2x.shape', x.shape)
            print('2z.shape', z.shape)
            x = x + z
            x = self.norm2(x)
            print('3x.shape', x.shape)
            print('3z.shape', z.shape)
            x = F.elu(x)

        x = self.lin0(x)
        x = F.elu(x)
        x = self.lin1(x)

        return x
        # output = {
        #             'logits': x[:, :21],
        #             'bfactor': x[:, 21:22],
        #             'ss3': x[:, 22:26],
        #             'ss8': x[:, 26:35],
        #             'rsa': x[:, 35:36]
        #         }
        # return output