import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import MLP, GATConv
import math

def SinusoidalPosEmb(x, num_steps, dim,rescale=4):
    x = x / num_steps * num_steps*rescale
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

class Relative_PositionalEmbedding(torch.nn.Module):
    def __init__(self, demb):
        super(Relative_PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class denoising_model(torch.nn.Module):
    def __init__(self, nlabel, nfeat, num_nodes, num_layers, num_linears, nhid, nhead=6, cat_mode=False, skip=False, types='continuous'):
        super(denoising_model, self).__init__()

        self.nfeat = nfeat
        self.num_nodes = num_nodes
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.cat_mode = cat_mode
        self.nhead = nhead
        self.skip = skip
        self.layers = torch.nn.ModuleList()
        self.time_mlp = nn.Sequential(
            nn.Linear(self.nhid, self.nhid * 2),
            nn.ELU(),
            nn.Linear(self.nhid * 2, self.nhid)
        )
        if self.cat_mode:
            self.time_mlp2 = nn.Linear(self.nhid, self.nhid*self.nhead)

        self.pos_emb = Relative_PositionalEmbedding(self.nfeat + self.nlabel)
        self.r_net = torch.nn.Linear(self.nfeat + self.nlabel + 1, self.nhead * self.nhid, bias=False)

        for i in range(self.depth):
            if i == 0:
                if self.cat_mode:
                    self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True))
                else:
                    self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=False))

            else:
                if i==self.depth-1:
                    if self.cat_mode:
                        self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=False))
                    else:
                        self.layers.append(GATConv(self.nhid+self.nlabel, self.nhid, nhead, concat=False))
                else:
                    if self.cat_mode:
                        self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True))
                    else:
                        self.layers.append(GATConv(self.nhid+self.nlabel, self.nhid, nhead, concat=False))


        self.fdim = self.nhid+self.nlabel
        self.activation = torch.nn.ELU()
        self.types = types
                
        if self.skip:
            self.lin_list = torch.nn.ModuleList()
            for i in range(0, self.depth):
                if self.cat_mode:
                    if i == 0:
                        self.lin_list.append(nn.Linear(self.nfeat+self.nlabel, self.nhid*self.nhead))
                    elif i == self.depth -1:
                        self.lin_list.append(nn.Linear(self.nhid*self.nhead+self.nlabel, self.nhid))
                    else:
                        self.lin_list.append(nn.Linear(self.nhid*self.nhead+self.nlabel, self.nhid*self.nhead))
                else:
                    if i == 0:
                        self.lin_list.append(nn.Linear(self.nfeat+self.nlabel, self.nhid))
                    else:
                        self.lin_list.append(nn.Linear(self.nhid+self.nlabel, self.nhid))
                    
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel*2, use_bn=False, activate_func=F.elu)
        self.final2 = nn.Linear(self.num_nodes * self.nlabel, self.nlabel)


    def forward(self, x, q_Y_sample, adj, t, num_steps):
        t_abs = SinusoidalPosEmb(t, num_steps, self.nhid)
        t_abs = self.time_mlp(t_abs)

        t_rel = self.pos_emb(t)
        t_rel = self.r_net(t_rel).view(-1, self.nhead, self.nhid)

        x = torch.cat([x, q_Y_sample], dim=-1)
        for i in range(self.depth):
            if self.cat_mode:
                if i == self.depth-1:
                    x_before_act = self.layers[i](x, adj, t_rel) + t_abs
                else:
                    x_before_act = self.layers[i](x, adj, t_rel) + self.time_mlp2(t_abs)
            else:
                x_before_act = self.layers[i](x, adj, t_rel) + t_abs
            if self.skip:
                x_before_act = x_before_act + self.lin_list[i](x)

            x = self.activation(x_before_act)
            x = torch.cat([x, q_Y_sample], dim=-1)

        updated_y = self.final(x).view(q_Y_sample.shape[0], -1)

        return updated_y

