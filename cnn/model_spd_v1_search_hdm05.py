import torch
import torch.nn as nn
import torch.nn.functional as F
from operations_spd_v1_hdm05 import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_SPDNet_v1 as PRIMITIVES
from genotypes import Genotype
import nn as nn_spd
import functional
import cplx.nn as nn_cplx


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride, type, reduction):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            if primitive.endswith(type):
                #print(primitive)
                if reduction:
                    if type == "reduced":
                        op = OPS[primitive](C_in, C_out, dim_in, dim_out,
                                            stride)
                        if 'Pool' in primitive:
                            op = nn.Sequential(op,
                                               nn_spd.BatchNormSPD(dim_out))
                    if type == "normal":
                        op = OPS[primitive](C_in, C_in, dim_out, dim_out,
                                            stride)
                        if 'Pool' in primitive:
                            op = nn.Sequential(op,
                                               nn_spd.BatchNormSPD(dim_out))
                else:
                    op = OPS[primitive](C_in, C_out, dim_in, dim_out, stride)
                    if 'Pool' in primitive:
                        op = nn.Sequential(op, nn_spd.BatchNormSPD(dim_out))
                self._ops.append(op)

    def forward(self, x, weights):
        #for op in self._ops:
        #    print("Operation",op)
        #    print("In shape",x.shape)
        #    s=op(x)
        #    print("Out_shape",s.shape)
        ops = torch.cat([op(x).unsqueeze(0) for op in self._ops], dim=0)
        weights_batched = weights.repeat(x.shape[0], 1)
        FM = []

        for i in range(ops[0].shape[1]):
            set_of_spds = torch.cat(
                [ops[j, :, i, :, :].unsqueeze(0) for j in range(ops.shape[0])],
                dim=0)
            s = set_of_spds.transpose(0, 1)
            #print(x.shape)
            #print(weights_batched.shape)
            FM.append(
                functional.bary_geom_weightedbatch(s.double(),
                                                   weights_batched.double()))
        FM = torch.cat(FM, dim=1)
        return FM


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, dim_in,
                 dim_out, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduceSPDNet(C_prev_prev, C_prev * 2,
                                                      dim_in, dim_out)  #check
        else:
            self.preprocess0 = ReLUConvBNSPDNet(C_prev_prev, C * 2,
                                                dim_in)  #check
        #print(C_prev)
        #print(C)
        #print(dim_in)
        if reduction_prev:
            self.preprocess1 = ReLUConvBNSPDNet(2 * C_prev, C_prev * 2,
                                                dim_out)
        else:
            self.preprocess1 = ReLUConvBNSPDNet(2 * C_prev, C * 2, dim_in)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        #self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                if reduction and j < 2:
                    stride = 3
                    type = "reduced"
                    C_in = C * 2
                    C_out = C * 2
                elif not reduction and j >= 2:
                    stride = 1
                    type = "normal"
                    C_in = C
                    C_out = C
                    dim_out = 30
                    dim_in = dim_out
                else:
                    stride = 1
                    type = "normal"
                    C_in = C * 2
                    C_out = int(C)
                op = MixedOp(C_in, C_out, dim_in, dim_out, stride, type,
                             reduction)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        #print(s0.shape)
        #print(s1.shape)
        s1 = self.preprocess1(s1)
        #print(s1.shape)
        states = [s0, s1]
        offset = 0
        #print(self._ops)
        #print(s0.shape)
        #print(s1.shape)
        for i in range(self._steps):
            s = torch.cat([
                self._ops[offset + j](h, weights[offset + j]).unsqueeze(0)
                for j, h in enumerate(states)
            ],
                          dim=0)
            UFM = []
            weights_batched = torch.tensor(1 / s.shape[0]).repeat(
                s.shape[0]).repeat(s[0].shape[0], 1)
            for i in range(s[0].shape[1]):
                set_of_spds = torch.cat(
                    [s[j, :, i, :, :].unsqueeze(0) for j in range(s.shape[0])],
                    dim=0)
                set_of_spds = set_of_spds.transpose(0, 1)
                UFM.append(
                    functional.bary_geom_weightedbatch(set_of_spds,
                                                       weights_batched))
            UFM = torch.cat(UFM, dim=1)
            offset += len(states)
            states.append(UFM)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self,
                 C,
                 dim_in,
                 dim_out,
                 num_classes,
                 layers,
                 criterion,
                 steps=2,
                 multiplier=2,
                 stem_multiplier=1):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._multiplier = multiplier
        C_curr = stem_multiplier * C
        window_size = 20
        hop_length = 10
        #self.split = nn_cplx.SplitSignal_cplx(2, window_size, hop_length)
        #self.covpool = nn_cplx.CovPool_cplx()
        self.stem = nn.Sequential(nn_spd.BiMap(1, 1, dim_in, dim_in),
                                  nn_spd.BatchNormSPD(dim_in))
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i % 2 == 0:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                C_curr *= 2
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, dim_in,
                        dim_out, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.logeig = nn_spd.LogEig()
        self.classifier = nn.Linear(8 * 30 * 30, num_classes)  # Check!

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self.dim_in, self.dim_out,
                            self._num_classes, self._layers,
                            self._criterion)  #.cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        model_new.double()
        return model_new

    def forward(self, input):
        #print(input.shape)
        #input = self.split(input)
        #print(input.shape)
        #input = self.covpool(input)
        #print(input.shape)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.logeig(s1)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        logits = self.classifier(out.view(out.size(0), -1))
        #print(logits)
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops_normal = 7
        num_ops_reduce = 7
        self.alphas_normal = torch.nn.Parameter(data=1e-3 *
                                                torch.randn(k, num_ops_normal),
                                                requires_grad=True)  #cuda
        self.alphas_reduce = torch.nn.Parameter(data=1e-3 *
                                                torch.randn(k, num_ops_reduce),
                                                requires_grad=True)  #cuda
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        #print(self._arch_parameters[0].shape)
        return self._arch_parameters

    def genotype(self):
        def _parse(weights, type):
            gene = []
            n = 2
            start = 0
            if type == "n":
                P = PRIMITIVES[0:7]
            if type == "r":
                P = PRIMITIVES[7:14]
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                if type == "n":
                    edges = sorted(
                        range(i + 2),
                        key=lambda x: -max(W[x][k] for k in range(len(W[x])) if
                                           (k != P.index('none_normal'))))[:2]
                else:
                    edges = sorted(
                        range(i + 2),
                        key=lambda x: -max(W[x][k] for k in range(len(W[x])) if
                                           (k != P.index('none_reduced'))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if type == "n":
                            if (k != P.index('none_normal')):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if (k != P.index('none_reduced')):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                    gene.append((P[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), 'n')
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), 'r')

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal,
                            normal_concat=concat,
                            reduce=gene_reduce,
                            reduce_concat=concat)
        return genotype
