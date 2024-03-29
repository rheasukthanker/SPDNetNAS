import torch
import torch.nn as nn

from utils import drop_path
from operations_spd_v0_afew import *

import nn as nn_spd
import functional


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, dim_in, dim_out,
                 reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduceSPDNet(C_prev_prev, C_prev,
                                                      dim_in, dim_out)  #check
        else:
            self.preprocess0 = ReLUConvBNSPDNet(C_prev_prev, C, dim_in,
                                                dim_in)  #check
        if reduction_prev:
            self.preprocess1 = ReLUConvBNSPDNet(C_prev, C_prev, dim_out,
                                                dim_out)
        else:
            self.preprocess1 = ReLUConvBNSPDNet(C_prev, C, dim_in, dim_in)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, dim_in, dim_out, op_names, indices, concat, reduction)

    def _compile(self, C, dim_in, dim_out, op_names, indices, concat,
                 reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        ind = {k: i for i, k in enumerate(OPS.keys())}
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            primitive = name
            if reduction and index < 2:
                stride = 2
                type = "reduced"
            else:
                stride = 1
                type = "normal"
            if reduction:
                if type == "reduced":
                    op = OPS[primitive](C, dim_in, dim_out, stride)
                    if 'Pool' in primitive:
                        op = nn.Sequential(op, nn_spd.BatchNormSPD(dim_out))
                if type == "normal":
                    index_normal = ind[primitive]
                    index_normal = index_normal - 6
                    primitive_new = list(ind.keys())[list(
                        ind.values()).index(index_normal)]
                    #print(primitive_new)
                    op = OPS[primitive_new](C, dim_out, dim_out, stride)
                    if 'Pool' in primitive_new:
                        op = nn.Sequential(op, nn_spd.BatchNormSPD(dim_out))
            else:
                op = OPS[primitive](C, dim_in, dim_out, stride)
                if 'Pool' in primitive:
                    op = nn.Sequential(op, nn_spd.BatchNormSPD(dim_out))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            ##DO WE DROP PATHS?
            #if self.training and drop_prob > 0.:
            #    if not (isinstance(op1, Skip_normal) or isinstance(op1,Skip_2)):
            #        h1 = drop_path(h1, drop_prob)
            #    if not (isinstance(op2, Skip_normal) or isinstance(op2,Skip_2)):
            #        h2 = drop_path(h2, drop_prob)
            s = torch.cat([h1.unsqueeze(0), h2.unsqueeze(0)], dim=0)
            #s= s.transpose(0,1)
            #print(s.shape)
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
            s = UFM
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):

    def __init__(self, C, dim_in, dim_out, num_classes, layers, genotype):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        #self._criterion = criterion
        #self._steps = steps
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._multiplier = stem_multiplier = 2
        C_curr = stem_multiplier * C
        #window_size = 20
        #hop_length = 10
        #self.split = nn_cplx.SplitSignal_cplx(2, window_size, hop_length)
        #self.covpool = nn_cplx.CovPool_cplx()
        self.fc_proj = nn.Linear(512, 100)
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
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, dim_in, dim_out,
                        reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, 2 * C_curr
        self.logeig = nn_spd.LogEig()
        self.classifier = nn.Linear(8 * 50 * 50, num_classes)  # Check

    def forward(self, input):
        #print(input.shape)
        #input = self.split(input)
        #print(input.shape)
        #input = self.covpool(input)
        #print(input.shape)
        input = torch.transpose(input, 1, 2)
        #print(input.shape)
        batch_size = input.shape[0]
        channels = input.shape[1]
        input = input.reshape([input.shape[0] * input.shape[1], 512])
        input = self.fc_proj(input)
        #current_frame = input
        current_frame = input.reshape([batch_size, channels, 100])
        #current_frame = current_frame.view(current_frame.shape[0], -1)
        current_frame = current_frame - torch.mean(current_frame,
                                                   dim=2).unsqueeze(2)
        factor = 1.0 / (current_frame.shape[1] - 1)
        current_framet = torch.transpose(current_frame, 1, 2)
        feature_covariance = factor * current_framet.matmul(
            current_frame).squeeze()
        lam = 10e-6 * torch.einsum('bii->b', feature_covariance)
        lam = lam.view(lam.shape[0], 1, 1)
        feature_covariance = feature_covariance + lam * torch.eye(100).repeat(
            lam.shape[0], 1, 1)
        #print(feature_covariance.shape)
        # print(lam.shape)
        #covs.append(feature_covariance.unsqueeze(1))
        #covs = torch.cat(covs, dim=1)
        s0 = s1 = self.stem(feature_covariance.unsqueeze(1))
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.logeig(s1)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
