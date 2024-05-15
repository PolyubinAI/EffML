""" Operations """
import torch
import torch.nn as nn
import genotypes as gt
import torch.nn.functional as F
import torch.nn as nn
from models import layers


OPS = {
    'none': lambda in_size, out_size, dropout, affine: Zero(in_size, out_size),
    'MLP_Relu': lambda in_size, out_size, dropout, affine: MLP(in_size, out_size, nn.ReLU(), dropout, affine),
    'MLP_Tanh': lambda in_size, out_size, dropout, affine: MLP(in_size, out_size, nn.Tanh(), dropout, affine),
    'MLP_Sigmoid': lambda in_size, out_size,dropout, affine: MLP(in_size, out_size, nn.Sigmoid(), dropout, affine),
    'MLP_Gelu':  lambda in_size, out_size,dropout, affine: MLP(in_size, out_size, nn.GELU(), dropout, affine),
    'skip_connect': lambda in_size, out_size, dropout, affine: nn.Identity() if in_size == out_size else nn.Linear(in_size, out_size),
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x

class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, in_size, out_size, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.mlp = MLP(in_size, out_size, activation=nn.ReLU())
        self.bn = nn.BatchNorm1d(out_size, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = self.mlp(x)
        out = self.bn(out)

        return out
    
class TabDLModel(nn.Module):
    def __init__(self, n_cont_features, cat_cardinalities, mlp_kwargs, model_type, activation):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        d_cat = sum(cat_cardinalities)
        self.activation = activation

        
        if model_type == 'MLP-PLR':
            d_embedding = 24
            self.cont_embeddings = layers.PeriodicEmbeddings(n_cont_features, d_embedding, lite=False)
            d_num = n_cont_features * d_embedding
        elif model_type == 'MLP':
            self.cont_embeddings = nn.Identity()
            d_num = n_cont_features
        
        self.backbone = layers.MLP(d_in=d_num + d_cat, **mlp_kwargs)

    def forward(self, x_cont, x_cat = None):
        x = []
        x.append(self.cont_embeddings(x_cont).flatten(1))
        # if x_cat is not None:
        #     x.extend(
        #         F.one_hot(column, cardinality)
        #         for column, cardinality in zip(x_cat.T, self.cat_cardinalities)
        #     )
        x = torch.column_stack(x)
        x = self.backbone(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size, out_size, activation, dropout=False, affine=True):
        super().__init__()
        #print(out_size, in_size)
        if dropout:
            self.net = nn.Sequential(
            nn.Linear(in_size, out_size),
            activation,
            # nn.BatchNorm1d(out_size, affine=affine)
            )
        else:
            self.net = nn.Sequential(
            nn.Linear(in_size, out_size),
            activation,
            nn.Dropout(),
            # nn.BatchNorm1d(out_size, affine=affine)
            )

    def forward(self, x):
        return self.net(x)



class Zero(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.out_size = out_size

    def forward(self, x):
            # return x * 0.
            device = torch.device('cuda:1')
            return nn.Parameter(torch.zeros((x.shape[0], self.out_size))).to(device)


# class MixedOp(nn.Module):
#     """ Mixed operation """
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self._ops = nn.ModuleList()
#         # self._lin = nn.ModuleList()
#         for primitive in gt.PRIMITIVES:
#             op = OPS[primitive](in_size, out_size, dropout=True, affine=False)
#             # self._lin.append(nn.Linear(, in_size))
#             self._ops.append(op)
#     def forward(self, x, weights):
#         """
#         Args:
#             x: input
#             weights: weight for each operation
#         """
#         y = torch.zeros_like(weights[0])
#         for op, lin in zip(self._ops, self._lin):
#             print("x shape", x.shape)
#             print("op shape", op)
#             print("lin shape", lin) 
#             # print('aaaaa', op)
#             l = lin(x)
#             print(f'l size {l.shape}')
#             o = op(l)
#             print(f'o size {o.shape}')
#             y = y + o
#             print("y shape", y.shape)
#             # print('bbbbb', op)
#         # # return sum(w * op(x) for w, op in zip(weights, self._ops))
#         return y

class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, in_size, out_size,):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](in_size, out_size, dropout=True, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        device = torch.device("cuda:1")

        return sum(w.to(device) * op(x.to(device)) for w, op in zip(weights, self._ops))