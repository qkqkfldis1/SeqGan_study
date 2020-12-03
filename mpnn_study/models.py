import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="8"

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


import torch
import torch.nn.functional as F
import torch.nn as nn
import time




class MPNN(torch.nn.Module):
    def __init__(self, n_edge_feature, n_atom_feature):
        super(MPNN, self).__init__()

        self.W = nn.Sequential(
            nn.Linear(n_atom_feature, 3 * n_atom_feature),
            nn.ReLU(),
            nn.Linear(3 * n_atom_feature, 1 * n_atom_feature),
            nn.ReLU(),
            nn.Linear(n_atom_feature, 3 * n_atom_feature),
            nn.ReLU(),
            nn.Linear(3 * n_atom_feature, 1 * n_atom_feature),
        )
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            # nn.Linear(n_feature, 3*n_feature),
            # nn.ReLU(),
            # nn.Linear(3*n_feature, 1*n_feature),
            # nn.ReLU(),
            nn.Linear(1 * n_edge_feature, n_atom_feature * n_atom_feature),
            # nn.ReLU(),
        )

        # self.A = nn.Parameter(torch.zeros(size=(n_atom_feature, n_atom_feature)))

    def forward(self, edge, adj, x): # edge [8, 32, 32, 8], adj [8, 32, 32], x [8, 32, 4]
        x_input = x.clone()
        # x = F.relu(self.W(x))

        message_matrix = self.cal_message(edge) # [8, 32, 32, 16]
        message_matrix = message_matrix.view(edge.size(0), edge.size(1), \
                                             edge.size(2), x.size(-1), x.size(-1)) # [8, 32, 32, 4, 4]
        x_repeat = x.unsqueeze(1).repeat(1, x.size(1), 1, 1).unsqueeze(-2) # [8, 32, 32, 1, 4]

        message = torch.einsum('abcde,abcef->abcdf', (x_repeat, message_matrix)) # [8, 32, 32, 1, 4], [8, 32, 32, 4, 4] -> [8, 32, 32, 1, 4]
        message = message.squeeze() # [8, 32, 32, 4]

        # e = torch.einsum('ijl,ikl->ijk', (torch.matmul(x,self.A), x))
        # e = e + e.permute((0,2,1))
        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=-1)
        # attention = attention*adj
        attention = adj

        message = message * attention.unsqueeze(-1).repeat(1, 1, 1, message.size(-1))
        message = message.sum(2).squeeze() # [8, 32, 4]

        reshaped_message = message.view(-1, x.size(-1)) # [256, 4]
        reshaped_x = x.view(-1, x.size(-1)) # [256, 4]
        retval = self.C(reshaped_message, reshaped_x) # [256, 4]
        retval = retval.view(x.size(0), x.size(1), x.size(2)) # [8, 32, 4]
        return retval + x_input


class MPNN_edge(torch.nn.Module):
    def __init__(self, n_edge_feature, n_atom_feature):
        super(MPNN_edge, self).__init__()

        self.W = nn.Sequential(
            nn.Linear(n_atom_feature, 3 * n_atom_feature),
            nn.ReLU(),
            nn.Linear(3 * n_atom_feature, 1 * n_atom_feature),
            nn.ReLU(),
            nn.Linear(n_atom_feature, 3 * n_atom_feature),
            nn.ReLU(),
            nn.Linear(3 * n_atom_feature, 1 * n_atom_feature),
        )
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            nn.Linear(1 * n_edge_feature + 2 * n_atom_feature, n_edge_feature * 2),
            nn.ReLU(),
            nn.Linear(n_edge_feature * 2, n_edge_feature * 2),
            nn.ReLU(),
            nn.Linear(n_edge_feature * 2, n_edge_feature * 2),
            nn.ReLU(),
            nn.Linear(n_edge_feature * 2, n_edge_feature * 2),
            nn.ReLU(),
            nn.Linear(n_edge_feature * 2, n_edge_feature),
        )

    def forward(self, edge, adj, x): # edge [8, 32, 32, 8], # adj [8, 32, 32] # x [8, 32, 8]
        # edge = F.relu(self.W(edge))
        h1 = x.unsqueeze(1).repeat(1, x.size(1), 1, 1) # [8, 32, 32, 8]
        h2 = x.unsqueeze(2).repeat(1, 1, x.size(1), 1) # [8, 32, 32, 8]
        message = self.cal_message(torch.cat([h1, h2, edge], -1)) # [8, 32, 32, 24] -> [8, 32, 32, 8] same with edge

        reshaped_message = message.view(-1, edge.size(-1)) # [8192, 8]
        reshaped_edge = edge.view(-1, edge.size(-1)) # [8192, 8]
        retval = self.C(reshaped_message, reshaped_edge) # [8192, 8]
        retval = retval.view(edge.size(0), edge.size(1), edge.size(2), edge.size(3)) # [8, 32, 32, 8]
        return retval


class IntraNet(torch.nn.Module):
    def __init__(self, n_atom_feature, n_edge_feature):
        super(IntraNet, self).__init__()

        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            nn.Linear(n_atom_feature * 2 + n_edge_feature, n_atom_feature),
        )

    def forward(self, edge, adj, x):
        h1 = x.unsqueeze(1).repeat(1, x.size(1), 1, 1)
        h2 = x.unsqueeze(2).repeat(1, 1, x.size(1), 1)

        concat = torch.cat([h1, h2, edge], -1)
        message = self.cal_message(concat)
        message = message * adj.unsqueeze(-1).repeat(1, 1, 1, message.size(-1))
        message = message.sum(2).squeeze()

        reshaped_message = message.view(-1, x.size(-1))
        reshaped_x = x.view(-1, x.size(-1))
        retval = self.C(reshaped_message, reshaped_x)
        retval = retval.view(x.size(0), x.size(1), x.size(2))

        return retval




class mpnn(torch.nn.Module):
    def __init__(self, args):
        super(mpnn, self).__init__()
        if args.dim_gnn % args.n_tower != 0:
            print('dim of gnn must be times of number of tower!!!')
            exit(-1)
        self.args = args
        num_filter = int(10 / args.filter_spacing) + 1
        self.filter_center = torch.Tensor([args.filter_spacing * i for i \
                                           in range(num_filter)])
        self.filter_gamma = args.filter_gamma
        self.node_embedding = nn.Sequential(nn.Embedding(4, args.dim_gnn),
                                            nn.Linear(args.dim_gnn, args.dim_gnn, bias=False)
                                            )
        self.edge_embedding = nn.Linear(num_filter, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList([nn.ModuleList([MPNN(args.dim_gnn, args.dim_gnn // args.n_tower) \
                                                   for j in range(self.args.n_tower)]) \
                                    for i in range(args.n_gnn)])
        self.gconv_edge = nn.ModuleList([MPNN_edge(args.dim_gnn, args.dim_gnn) for i in range(args.n_gnn)])

        self.concat = nn.ModuleList([nn.Linear(args.dim_gnn, args.dim_gnn) \
                                     for i in range(args.n_gnn)])

        self.linear = nn.Sequential(
            nn.Linear(args.dim_gnn * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(args.dim_gnn * 3, self.args.n_tower, 4 ** 2), args.n_trans)

    #        self.dummy_atom = nn.Parameter(torch.zeros(args.dim_gnn))
    #        self.dummy_edge = nn.Parameter(torch.zeros(args.dim_gnn))
    def forward(self, H, A, edge): # H: [8, 32], # A: [8, 32, 32] # edge: [8, 32, 32]
        #        for p in self.node_embedding.parameters():
        #            print(H.device, p.device)
        h = self.node_embedding(H) # (8, 32, 8)
        edge = edge.unsqueeze(-1).repeat(1, 1, 1, self.filter_center.size(-1)) # [8, 32, 32, 21]
        filter_center = self.filter_center.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(edge.device) # [21] -> [1, 1, 1, 21]

        edge = torch.exp(-torch.pow(edge - filter_center, 2) * self.filter_gamma)
        e = self.edge_embedding(edge) # [8, 32, 32, 8]
        # dummy_atom = self.dummy_atom.unsqueeze(0).repeat(h.size(0), 1)
        # dummy_atom = dummy_atom.unsqueeze(1)
        # h = torch.cat([dummy_atom,h], 1)

        # dummy_edge1 = self.dummy_edge.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dummy_edge2 = self.dummy_edge.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # dummy_edge1 = dummy_edge1.repeat(e.size(0),1,e.size(1),1)
        # dummy_edge2 = dummy_edge2.repeat(e.size(0),e.size(1)+1,1,1)
        # e = torch.cat([dummy_edge1,e], 1)
        # e = torch.cat([dummy_edge2,e], 2)

        for i in range(len(self.gconv)):
            e = self.gconv_edge[i](e, A, h) # e: [8, 32, 32, 8]

            hs = torch.split(h, self.args.dim_gnn // self.args.n_tower, -1) # hs ([8, 32, 4], [8, 32, 4])
            hs_new = []
            for j in range(len(hs)):
                hs_new.append(self.gconv[i][j](e, A, hs[j]))

            h = torch.cat(hs_new, -1)
            h = self.concat[i](h)

        h1 = h.unsqueeze(1).repeat(1, h.size(1), 1, 1)
        h2 = h.unsqueeze(2).repeat(1, 1, h.size(1), 1)
        retval = torch.cat([h1, h2, e], -1)
        # print(retval[0], A[0])
        retval = self.transformer(retval.view(retval.shape[0], -1, retval.shape[-1]).permute(1, 0, 2),
                                  src_key_padding_mask=(A.view(retval.shape[0], -1) == 0)).permute(1, 0, 2)

        retval = self.linear(retval)
        retval = retval.reshape((retval.size(0), h.size(1), h.size(1), retval.size(-1)))
        A = A.unsqueeze(-1).repeat(1, 1, 1, retval.shape[-1])
        # print((retval*A)[0,:,:,0]);exit(-1)
        retval = (retval * A).sum(dim=[-2, -3])
        retval = retval / A.sum(dim=[-2, -3])
        # retval*=A.view(retval.shape[0],-1,1)
        # retval = retval.sum(dim=[-1,-2]) /A.sum(dim=[-1,-2])

        return retval
class config:
    dim_gnn = 8
    n_tower = 2
    filter_spacing = 0.5
    filter_gamma = 1
    n_gnn = 2
    n_trans = 16

if __name__ == "__main__":
    args = config
    
    model = mpnn(args).cuda()

    edge = torch.rand(8, 32, 32).cuda()
    H = torch.ones(8, 32).type(torch.long).cuda()
    A = torch.rand(8, 32, 32).cuda()

    result = model(H, A, edge)