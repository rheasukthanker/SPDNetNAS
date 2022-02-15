import torch
import nn as nn_spd
import torch.nn as nn
import functional

OPS = {
    'none_normal':
    lambda C_in, dim_in, dim_out, stride: Zero_normal(dim_out),
    'WeightedPooling_normal':
    lambda C_in, dim_in, dim_out, stride: WeightedPooling(C_in),
    'BiMap_0_normal':
    lambda C_in, dim_in, dim_out, stride: BiMap_0_normal(C_in, dim_out),
    'BiMap_1_normal':
    lambda C_in, dim_in, dim_out, stride: BiMap_1_normal(C_in, dim_out),
    'BiMap_2_normal':
    lambda C_in, dim_in, dim_out, stride: BiMap_2_normal(C_in, dim_out),
    'Skip_normal':
    lambda C_in, dim_in, dim_out, stride: Skip_normal(),
    'none_reduced':
    lambda C_in, dim_in, dim_out, stride: Zero_reduced(C_in, dim_in, dim_out),
    'AvgPooling2_reduced':
    lambda C_in, dim_in, dim_out, stride: AvgPooling_2(dim_in, dim_out, stride
                                                       ),
    'MaxPooling_reduced':
    lambda C_in, dim_in, dim_out, stride: MaxPooling(dim_in, dim_out, stride),
    'BiMap_1_reduced':
    lambda C_in, dim_in, dim_out, stride: BiMap_1_reduced(
        C_in, dim_in, dim_out),
    'BiMap_2_reduced':
    lambda C_in, dim_in, dim_out, stride: BiMap_2_reduced(
        C_in, dim_in, dim_out),
    'Skip_reduced':
    lambda C_in, dim_in, dim_out, stride: Skip_2(C_in, dim_in, dim_out),
}


#'BiMap_0_reduced': lambda C_in, dim_in,dim_out, stride: BiMap_0_reduced(C_in,dim_in,dim_out),
class ReLUConvBNSPDNet(nn.Module):

    def __init__(self, C_in, C_out, dim_in, dim_out):
        super(ReLUConvBNSPDNet, self).__init__()
        self.op = nn.Sequential(nn_spd.ReEig(),
                                nn_spd.BiMap(C_out, C_in, dim_in, dim_out),
                                nn_spd.BatchNormSPD(dim_out))

    def forward(self, x):
        #print(torch.max(x))
        return self.op(x)


class FactorizedReduceSPDNet(nn.Module):

    def __init__(self, C_in, C_out, dim_in, dim_out):
        super(FactorizedReduceSPDNet, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn_spd.ReEig()
        #print(C_out,C_in,dim_in,dim_out)
        self.conv_1 = nn_spd.BiMap(C_out, C_in, dim_in, dim_out)
        #self.conv_2 = nn_spd.BiMap(C_out, C_in,dim_in,dim_out)
        self.bn = nn_spd.BatchNormSPD(dim_out)

    def forward(self, x):
        #print(x.shape)
        x = self.relu(x)
        out = self.conv_1(x)
        out = self.bn(out)
        return out


class Zero_normal(nn.Module):

    def __init__(self, dim_out):
        super(Zero_normal, self).__init__()
        self.dim_out = dim_out

    def forward(self, x):
        output = x.mul(0.)
        output_diag = torch.diag(torch.tensor(1).repeat(self.dim_out))
        output = output + output_diag
        return output


class Zero_reduced(nn.Module):

    def __init__(self, C_in, dim_in, dim_out):
        super(Zero_reduced, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.C_in = C_in
        self.C_out = C_in

    def forward(self, x):
        output = torch.zeros(x.shape[0],
                             self.C_out,
                             self.dim_out,
                             self.dim_out,
                             dtype=torch.float64)
        output_diag = torch.diag(torch.tensor(1).repeat(self.dim_out))
        output = output + output_diag
        return output


class WeightedPooling(nn.Module):

    def __init__(self, C_in):
        super(WeightedPooling, self).__init__()
        self.C_in = C_in
        self.weights = torch.nn.Parameter(data=torch.Tensor(
            self.C_in, self.C_in),
                                          requires_grad=True)
        self.weights.data.uniform_(0, 1)

    def forward(self, x):
        weighted_channels = []

        weights_batched = self.weights.softmax(1).repeat(x.shape[0], 1, 1)
        #print(weights_batched.shape)
        for i in range(0, self.C_in):
            weighted_channels.append(
                functional.bary_geom_weightedbatch(
                    x, weights_batched[:, i, :].squeeze(1)))
        #print(weighted_channels[0].shape)
        output = torch.cat(weighted_channels, dim=1)
        return output


class BiMap_0_normal(nn.Module):

    def __init__(self, C_in, dim_in):
        super(BiMap_0_normal, self).__init__()
        self.C_in = C_in
        self.dim_in = dim_in
        self.layers = nn.Sequential(
            nn_spd.BiMap(self.C_in, self.C_in, self.dim_in, self.dim_in),
            nn_spd.BatchNormSPD(self.dim_in))

    def forward(self, x):
        output = self.layers(x)
        return output


class BiMap_1_normal(nn.Module):

    def __init__(self, C_in, dim_in):
        super(BiMap_1_normal, self).__init__()
        self.C_in = C_in
        self.dim_in = dim_in
        self.layers = nn.Sequential(
            nn_spd.BiMap(self.C_in, self.C_in, self.dim_in, self.dim_in),
            nn_spd.BatchNormSPD(self.dim_in), nn_spd.ReEig())

    def forward(self, x):
        output = self.layers(x)
        return output


class BiMap_2_normal(nn.Module):

    def __init__(self, C_in, dim_in):
        super(BiMap_2_normal, self).__init__()
        self.C_in = C_in
        self.dim_in = dim_in
        self.layers = nn.Sequential(
            nn_spd.ReEig(),
            nn_spd.BiMap(self.C_in, self.C_in, self.dim_in, self.dim_in),
            nn_spd.BatchNormSPD(self.dim_in))

    def forward(self, x):
        output = self.layers(x)
        return output


class Skip_normal(nn.Module):

    def __init__(self):
        super(Skip_normal, self).__init__()

    def forward(self, x):
        return x


class AvgPooling_2(nn.Module):

    def __init__(self, dim_in, dim_out, stride):
        super(AvgPooling_2, self).__init__()
        self.stride = stride
        #OPTIONS
        #pool_base = torch.nn.AvgPool2d(155, stride=5)
        self.dim_in = dim_in
        self.dim_out = dim_out
        #pool_base = torch.nn.AvgPool2d(251, stride=3)
        if dim_in == 400 and dim_out == 50:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       torch.nn.AvgPool2d(3, stride=8),
                                       nn_spd.ExpEig())
        elif dim_in == 400 and dim_out == 100:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       torch.nn.AvgPool2d(3, stride=4),
                                       nn_spd.ExpEig())
        elif dim_in == 100 and dim_out == 50:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       torch.nn.AvgPool2d(2, stride=2),
                                       nn_spd.ExpEig())
        elif dim_in == 400 and dim_out == 200:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       torch.nn.AvgPool2d(2, stride=2),
                                       nn_spd.ExpEig())
        elif dim_in == 200 and dim_out == 100:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       torch.nn.AvgPool2d(2, stride=2),
                                       nn_spd.ExpEig())
        elif dim_in == 400 and dim_out == 300:
            self.log_eig = nn_spd.LogEig()
            self.pool = torch.nn.AvgPool2d(2, stride=2)
            self.exp = nn_spd.ExpEig()
            #self.model = nn.Sequential(nn_spd.LogEig(), torch.nn.AvgPool2d(2, stride=2), nn_spd.ExpEig())
        elif dim_in == 300 and dim_out == 200:
            self.log_eig = nn_spd.LogEig()
            self.pool = torch.nn.AvgPool2d(2, stride=2)
            self.exp = nn_spd.ExpEig()

    def forward(self, x):
        if self.dim_in == 300 or self.dim_out == 300:
            x = self.log_eig(x)
            x = torch.nn.functional.pad(x, [100, 100, 100, 100])
            x = self.pool(x)
            output = self.exp(x)
        else:
            output = self.model(x)
        return output


class MaxPooling(nn.Module):

    def __init__(self, dim_in, dim_out, stride):
        super(MaxPooling, self).__init__()
        self.stride = stride
        if dim_in == 400 and dim_out == 50:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       nn.MaxPool2d(3, stride=8),
                                       nn_spd.ExpEig())
        elif dim_in == 400 and dim_out == 100:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       torch.nn.MaxPool2d(3, stride=4),
                                       nn_spd.ExpEig())
        elif dim_in == 100 and dim_out == 50:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       nn.MaxPool2d(2, stride=2),
                                       nn_spd.ExpEig())
        elif dim_in == 400 and dim_out == 200:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       nn.MaxPool2d(2, stride=2),
                                       nn_spd.ExpEig())
        elif dim_in == 200 and dim_out == 100:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       nn.MaxPool2d(2, stride=2),
                                       nn_spd.ExpEig())
        elif dim_in == 400 and dim_out == 300:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       nn.MaxPool2d(101, stride=1),
                                       nn_spd.ExpEig())
        elif dim_in == 300 and dim_out == 200:
            self.model = nn.Sequential(nn_spd.LogEig(),
                                       nn.MaxPool2d(101, stride=1),
                                       nn_spd.ExpEig())

    def forward(self, x):
        output = self.model(x)
        return output


class BiMap_0_reduced(nn.Module):

    def __init__(self, C_in, dim_in, dim_out):
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


class BiMap_1_reduced(nn.Module):

    def __init__(self, C_in, dim_in, dim_out):
        super(BiMap_1_reduced, self).__init__()
        self.C_in = C_in
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.model = nn.Sequential(
            nn_spd.ReEig(),
            nn_spd.BiMap(self.C_in, self.C_in, self.dim_in, self.dim_out),
            nn_spd.BatchNormSPD(self.dim_out))

    def forward(self, x):
        output = self.model(x)
        return output


class BiMap_2_reduced(nn.Module):

    def __init__(self, C_in, dim_in, dim_out):
        super(BiMap_2_reduced, self).__init__()
        self.dim_in = dim_in
        self.C_in = C_in
        self.dim_out = dim_out
        self.model = nn.Sequential(
            nn_spd.BiMap(self.C_in, self.C_in, self.dim_in, self.dim_out),
            nn_spd.BatchNormSPD(self.dim_out), nn_spd.ReEig())

    def forward(self, x):
        output = self.model(x)
        return output


class Skip_2(nn.Module):

    def __init__(self, C_in, dim_in, dim_out):
        super(Skip_2, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.C_in = C_in
        self.C_out = C_in

    def concat_pairs(self, m1, m2):  # pytorch version >=1.4
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
        return output
