import torch
import nn as nn_spd
import torch.nn as nn
import functional

OPS = {
    'none_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: Zero_normal(
        C_in, C_out, dim_in, dim_out, stride),
    'AvgPooling_1_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: AvgPooling_1(
        C_in, C_out, dim_in, dim_out, stride),
    'DiMap_1_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: DiMap_1(
        C_in, C_out, dim_in, dim_out, stride),
    'DiMap_2_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: DiMap_2(
        C_in, C_out, dim_in, dim_out, stride),
    'BiMap_1_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: BiMap_1_normal(
        C_in, C_out, dim_in, dim_out, stride),
    'BiMap_2_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: BiMap_2_normal(
        C_in, C_out, dim_in, dim_out, stride),
    'Skip_1_normal':
    lambda C_in, C_out, dim_in, dim_out, stride: Skip_1(
        C_in, C_out, dim_in, dim_out, stride),
    'none_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: Zero_reduced(
        C_in, C_out, dim_in, dim_out, stride),
    'AvgPooling_2_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: AvgPooling_2(
        C_in, C_out, dim_in, dim_out, stride),
    'MaxPooling_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: MaxPooling(
        C_in, C_out, dim_in, dim_out, stride),
    'BiMap_0_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: BiMap_0_reduced(
        C_in, C_out, dim_in, dim_out, stride),
    'BiMap_1_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: BiMap_1_reduced(
        C_in, C_out, dim_in, dim_out, stride),
    'BiMap_2_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: BiMap_2_reduced(
        C_in, C_out, dim_in, dim_out, stride),
    'Skip_2_reduced':
    lambda C_in, C_out, dim_in, dim_out, stride: Skip_2(
        C_in, C_out, dim_in, dim_out, stride),
}


#    'BiMap_0': lambda C_in, C_out, dim_in, dim_out, factor, stride: BiMap_0(C_in, dim_in, factor),
#'MaxPooling_normal': lambda C_in,dim_in,factor,stride: MaxPooling(stride),
class FactorizedReduceSPDNet(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out):
        super(FactorizedReduceSPDNet, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn_spd.ReEig()
        self.conv_1 = nn_spd.BiMap(C_out, C_in, dim_in, dim_out)
        #self.conv_2 = nn_spd.BiMap(C_out, C_in,dim_in,dim_out)
        self.bn = nn_spd.BatchNormSPD(dim_out)

    def forward(self, x):
        x = self.relu(x)
        out = self.conv_1(x)
        out = self.bn(out)
        return out


class ReLUConvBNSPDNet(nn.Module):
    def __init__(self, C_in, C_out, dim_in):
        super(ReLUConvBNSPDNet, self).__init__()
        self.op = nn.Sequential(nn_spd.ReEig(),
                                nn_spd.BiMap(C_out, C_in, dim_in, dim_in),
                                nn_spd.BatchNormSPD(dim_in))

    def forward(self, x):
        return self.op(x)


class BiMap_1_normal(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(BiMap_1_normal, self).__init__()
        self.C_in = C_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.C_out = C_out
        if C_in == C_out:
            self.layers = nn.Sequential(
                nn_spd.BiMap(self.C_out, self.C_in, self.dim_in, self.dim_in),
                nn_spd.BatchNormSPD(self.dim_in), nn_spd.ReEig())
        else:
            self.layers = nn.Sequential(
                nn_spd.BiMap(self.C_out, self.C_in, self.dim_out,
                             self.dim_out), nn_spd.BatchNormSPD(self.dim_out),
                nn_spd.ReEig())

    def forward(self, x):
        output = self.layers(x)
        return output


class BiMap_2_normal(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(BiMap_2_normal, self).__init__()
        self.C_in = C_in
        self.dim_in = dim_in
        self.C_out = C_out
        self.dim_out = dim_out
        if C_in == C_out:
            self.layers = nn.Sequential(
                nn_spd.ReEig(),
                nn_spd.BiMap(self.C_out, self.C_in, self.dim_in, self.dim_in),
                nn_spd.BatchNormSPD(self.dim_in))
        else:
            self.layers = nn.Sequential(
                nn_spd.ReEig(),
                nn_spd.BiMap(self.C_out, self.C_in, self.dim_out,
                             self.dim_out), nn_spd.BatchNormSPD(self.dim_out))

    def forward(self, x):
        output = self.layers(x)
        return output


class BiMap_1_reduced(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(BiMap_1_reduced, self).__init__()
        self.model = nn.Sequential(nn_spd.ReEig(),
                                   nn_spd.BiMap(C_in, C_in, dim_in, dim_out),
                                   nn_spd.BatchNormSPD(dim_out))

    def forward(self, x):
        output = self.model(x)
        return output


class BiMap_2_reduced(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(BiMap_2_reduced, self).__init__()
        self.model = nn.Sequential(nn_spd.BiMap(C_in, C_in, dim_in, dim_out),
                                   nn_spd.BatchNormSPD(dim_out),
                                   nn_spd.ReEig())

    def forward(self, x):
        output = self.model(x)
        return output


class BiMap_0_reduced(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(BiMap_0_reduced, self).__init__()
        self.C_in = C_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.model = nn.Sequential(
            nn_spd.BiMap(self.C_in, self.C_in, self.dim_in, self.dim_out),
            nn_spd.BatchNormSPD(self.dim_out))

    def forward(self, x):
        output = self.model(x)
        return output


class MaxPooling(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(MaxPooling, self).__init__()
        self.stride = stride
        self.model = nn.Sequential(nn_spd.LogEig(),
                                   nn.MaxPool2d(4, stride=self.stride),
                                   nn_spd.ExpEig())

    def forward(self, x):
        output = self.model(x)
        return output


class Skip_1(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(Skip_1, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.C_in = C_in
        self.C_out = C_out
        self.factor = int(C_in / C_out)

    def concat(self, m):  #version  >1.4
        Uis, Dis, _ = torch.svd(m)
        if m.shape[1] == 2:
            U_1 = Uis[:, 0, :, :]
            U_2 = Uis[:, 1, :, :]
            r_1 = torch.cat((U_1, torch.zeros_like(U_1)), dim=1)
            r_2 = torch.cat((torch.zeros_like(U_2), U_2), dim=1)
            Ub = torch.cat((r_1, r_2), dim=2)
            D_1 = torch.diag_embed(Dis[:, 0, :])
            D_2 = torch.diag_embed(Dis[:, 1, :])
            r_1 = torch.cat((D_1, torch.zeros_like(D_1)), dim=1)
            r_2 = torch.cat((torch.zeros_like(D_2), D_2), dim=1)
            Db = torch.cat((r_1, r_2), dim=2)
            Cb = torch.matmul(torch.matmul(Ub, Db), torch.transpose(Ub, 1, 2))
            Cb = Cb.unsqueeze(1)
        else:
            Cb = m

        return Cb

    def forward(self, x):
        num_pairs = int(x.shape[1] / self.factor)
        concatenated = []
        end_index = 0
        for i in range(num_pairs):  # Vectorize
            start_index = end_index
            end_index = (i + 1) * (self.factor)
            Cb_pair = self.concat(x[:, start_index:end_index, :, :])
            concatenated.append(Cb_pair)
            #print(Cb_pair.shape)
        output = torch.cat(concatenated, dim=1)
        # Apply BiMap to downsample back large SPD
        if self.C_in == self.C_out:
            output = nn_spd.BiMap(self.C_out, self.C_out, output.shape[2],
                                  self.dim_in)(output)
        else:
            output = nn_spd.BiMap(self.C_out, self.C_out, output.shape[2],
                                  self.dim_out)(output)
        return output


class Skip_2(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(Skip_2, self).__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.C_in = C_in
        self.C_out = C_in

    def concat_pairs(self, m1, m2):  # version >=1.4
        U1, D1, _ = torch.svd(m1)
        U2, D2, _ = torch.svd(m2)
        r_1 = torch.cat((U1, torch.zeros_like(U1)), dim=1)
        r_2 = torch.cat((torch.zeros_like(U2), U2), dim=1)
        Ub = torch.cat((r_1, r_2), dim=2)
        D_1 = torch.diag_embed(D1)
        D_2 = torch.diag_embed(D2)
        r_1 = torch.cat((D_1, torch.zeros_like(D_1)), dim=1)
        r_2 = torch.cat((torch.zeros_like(D_2), D_2), dim=1)
        Db = torch.cat((r_1, r_2), dim=2)
        Cb = torch.matmul(torch.matmul(Ub, Db), torch.transpose(Ub, 1, 2))
        Cb = Cb.unsqueeze(1)
        return Cb

    def forward(self, x):
        concatenated = []
        half_out_dim = int(self.dim_out / 2)
        bimap_1 = nn_spd.BiMap(self.C_in, self.C_in, self.dim_in,
                               half_out_dim)(x)
        bimap_2 = nn_spd.BiMap(self.C_in, self.C_in, self.dim_in,
                               half_out_dim)(x)
        for i in range(self.C_in):  # Vectorize
            Cb_pair = self.concat_pairs(bimap_1[:, i, :, :], bimap_2[:,
                                                                     i, :, :])
            concatenated.append(Cb_pair)
        output = torch.cat(concatenated, dim=1)
        return output.double()


class Zero_normal(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(Zero_normal, self).__init__()
        self.dim_out = dim_out
        self.C_out = C_out
        self.C_in = C_in
        self.dim_in = dim_in

    def forward(self, x):
        batch_size = x.shape[0]
        #print(self.dim_in)
        #print(self.dim_out)
        if (self.C_in == self.C_out) & (self.dim_in == self.dim_out):
            zero_out = torch.zeros(batch_size, self.C_out, self.dim_in,
                                   self.dim_in)
        else:
            zero_out = torch.zeros(batch_size, self.C_out, self.dim_out,
                                   self.dim_out)
        output_diag = torch.diag(torch.tensor(1).repeat(zero_out.shape[-1]))
        output = zero_out + output_diag
        return output.double()


class Zero_reduced(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(Zero_reduced, self).__init__()
        self.dim_out = dim_out
        self.C_out = C_in

    def forward(self, x):
        batch_size = x.shape[0]
        zero_out = torch.zeros(batch_size, self.C_out, self.dim_out,
                               self.dim_out)
        output_diag = torch.diag(torch.tensor(1).repeat(zero_out.shape[-1]))
        output = zero_out + output_diag
        return output.double()


class DiMap_1(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(DiMap_1, self).__init__()
        self.C_in = C_in
        self.weight_1 = torch.nn.Parameter(data=torch.Tensor(2),
                                           requires_grad=True)
        self.weight_1.data.uniform_(0, 1)
        if C_in == C_out:
            self.bn = nn_spd.BatchNormSPD(dim_in)
        else:
            self.bn = nn_spd.BatchNormSPD(dim_out)
        self.reig = nn_spd.ReEig()
        self.num_pairs = C_out
        self.factor = int(C_in / C_out)

    def forward(self, x):

        pooled = []
        end_index = 0
        weights_batched = self.weight_1.softmax(0).repeat(x.shape[0], 1)
        for i in range(self.num_pairs):
            start_index = end_index
            end_index = (i + 1) * (self.factor)
            set_of_spds = x[:, start_index:end_index, :, :]
            pooled.append(
                functional.bary_geom_weightedbatch(set_of_spds,
                                                   weights_batched))
        output = torch.cat(pooled, dim=1)
        output = self.bn(output)
        output = self.reig(output)
        return output


class DiMap_2(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(DiMap_2, self).__init__()
        self.C_in = C_in
        self.weight_1 = torch.nn.Parameter(data=torch.Tensor(2),
                                           requires_grad=True)
        self.weight_1.data.uniform_(0.0, 1.0)
        if C_in == C_out:
            self.bn = nn_spd.BatchNormSPD(dim_in)
        else:
            self.bn = nn_spd.BatchNormSPD(dim_out)
        self.reig = nn_spd.ReEig()
        self.C_out = C_out

    def forward(self, x):
        pooled = []
        end_index = 0
        index_channel_2 = 0
        weights_batched = self.weight_1.softmax(0).repeat(x.shape[0], 1)
        #print(x.shape)
        while index_channel_2 < (x.shape[1] - 1):
            index_channel_1 = end_index
            index_channel_2 = end_index + 2
            set_of_spds = x[:, [index_channel_1, index_channel_2], :, :]
            a = functional.bary_geom_weightedbatch(set_of_spds,
                                                   weights_batched)
            pooled.append(a)  #replace with FM
            if self.C_in == self.C_out:
                pooled.append(a)
            if ((index_channel_2 + 1) % 4 == 0):
                end_index = end_index + 3
            else:
                end_index = end_index + 1
        output = torch.cat(pooled, dim=1)
        output = self.bn(output)
        output = self.reig(output)
        return output


class AvgPooling_2(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(AvgPooling_2, self).__init__()
        self.model = nn.Sequential(nn_spd.LogEig(),
                                   nn.AvgPool2d(4, stride=stride),
                                   nn_spd.ExpEig())

    def forward(self, x):
        output = self.model(x)
        return output


class AvgPooling_1(nn.Module):
    def __init__(self, C_in, C_out, dim_in, dim_out, stride):
        super(AvgPooling_1, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.factor = int(C_in / C_out)

    def forward(self, x):
        num_pairs = self.C_out
        #print(x.shape)
        pooled = []
        end_index = 0
        #print(self.factor)
        weights_batched = torch.tensor(1. / self.factor).repeat(
            self.factor).repeat(x.shape[0], 1)
        for i in range(num_pairs):
            start_index = end_index
            end_index = (i + 1) * (self.factor)
            #print(start_index)
            #print(end_index)
            set_of_spds = x[:, start_index:end_index, :, :]
            #print(set_of_spds.shape)
            pooled.append(
                functional.bary_geom_weightedbatch(set_of_spds,
                                                   weights_batched))
        output = torch.cat(pooled, dim=1)
        return output
