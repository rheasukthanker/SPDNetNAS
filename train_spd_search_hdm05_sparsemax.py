import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.autograd import Variable
from model_spdnet_v0_search_sparsemax import Network
from architect import Architect
import random
import torch as th
from optimizers import MixOptimizer
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser("hdm05")
parser.add_argument(
    '--data',
    type=str,
    default='data/hdm05/',
    help='location of the data corpus')  #../../code_brooks/data/hdm05/
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.025,
                    help='init learning rate')
parser.add_argument('--learning_rate_min',
                    type=float,
                    default=0.001,
                    help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay',
                    type=float,
                    default=3e-4,
                    help='weight decay')
parser.add_argument('--report_freq',
                    type=float,
                    default=1,
                    help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels',
                    type=int,
                    default=16,
                    help='num of init channels')
parser.add_argument('--layers',
                    type=int,
                    default=8,
                    help='total number of layers')
parser.add_argument('--model_path',
                    type=str,
                    default='saved_models',
                    help='path to save the model')
parser.add_argument('--cutout',
                    action='store_true',
                    default=False,
                    help='use cutout')
parser.add_argument('--cutout_length',
                    type=int,
                    default=16,
                    help='cutout length')
parser.add_argument('--drop_path_prob',
                    type=float,
                    default=0.3,
                    help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip',
                    type=float,
                    default=5,
                    help='gradient clipping')
parser.add_argument('--train_portion',
                    type=float,
                    default=0.5,
                    help='portion of training data')
parser.add_argument('--unrolled',
                    action='store_true',
                    default=True,
                    help='use one-step unrolled validation loss')  #false
parser.add_argument('--arch_learning_rate',
                    type=float,
                    default=3e-4,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',
                    type=float,
                    default=1e-3,
                    help='weight decay for arch encoding')
parser.add_argument('--penalty_parameter',
                    type=float,
                    default=1,
                    help='weight penalty hyperparameter of loss function')
parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu')
args = parser.parse_args()

args.save = 'search-NORMALv0HDM-lr_{}-{}'.format(args.learning_rate,
                                                 time.strftime("%m%d-%H%M"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(
    os.path.join(args.save, f'log-norm-{args.learning_rate}.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
data_path = 'data/hdm05/'  #'data/hdm05/'  # data path
pval = 0.125  # validation percentage
ptest = 0.5  # test percentage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.device = device
# Check whether GPU is loaded or not
if torch.cuda.is_available():
    print('Device : GPU')
else:
    print('Device : CPU')


def use_sparsemax_as_convex_layer(x, n):
    x_ = cp.Parameter(n)
    y_ = cp.Variable(n)
    obj = cp.sum_squares(x_ - y_)
    cons = [cp.sum(y_) == 1, 0. <= y_, y_ <= 1.]
    prob = cp.Problem(cp.Minimize(obj), cons)
    layer = CvxpyLayer(prob, [x_], [y_])
    y, = layer(x)
    return y


class DatasetHDM05(data.Dataset):

    def __init__(self, path, names):
        self._path = path
        self._names = names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        x = np.load(self._path + self._names[item])[None, :, :].real
        x = th.from_numpy(x).double()
        y = int(self._names[item].split('.')[0].split('_')[-1])
        y = th.from_numpy(np.array(y)).long()
        return x, y


class DataLoaderHDM05:

    def __init__(self, data_path, pval=0.125, ptest=0.5, trainperc=100):
        # trainperc = Percentage of total Nb of training samples, default is 100 % (37.5% of total dataset), choose accordingly (always <= 100)
        for filenames in os.walk(data_path):
            names = sorted(filenames[2])
        print(len(names))
        # We have to decide a seed to be consistent
        random.Random(args.seed).shuffle(names)
        N_val = int(pval * len(names))  # 25% of 50% -> 0.125% total -> 260
        N_test = int(ptest * len(names))  # 50 % total -> 1043
        N_train = int(
            trainperc * 0.01 *
            (len(names) - N_test - N_val))  # if trainperc = 100 -> 783

        test_set = DatasetHDM05(data_path,
                                names[:N_test])  # NOT USING IT FOR SEARCH
        val_set = DatasetHDM05(data_path, names[N_test:N_test + N_val])
        train_set = DatasetHDM05(
            data_path, names[N_val + N_test:N_val + N_test + N_train]
        )  # trainsize = Nb of training samples, default is 782 (37.5%)

        self._train_generator = data.DataLoader(train_set,
                                                batch_size=args.batch_size,
                                                shuffle='True')
        self._val_generator = data.DataLoader(val_set,
                                              batch_size=args.batch_size,
                                              shuffle='True')
        self._test_generator = data.DataLoader(
            test_set, batch_size=args.batch_size,
            shuffle='False')  # NOT USING IT FOR SEARCH


num_classes = 117


def main(trainperc):
    #if not torch.cuda.is_available():
    #  logging.info('no gpu device available')
    #  sys.exit(1)
    args.layers = 2
    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    #logging.info('gpu device = %d' % args.gpu)
    #logging.info("args = %s", args)
    logging.info("Device = {}".format(device))
    criterion = nn.CrossEntropyLoss().to(device)
    #criterion = criterion.cuda()
    model = Network(1, 93, 30, num_classes, 2, criterion)
    model = model.double().to(device)
    p = model.state_dict()
    #for params in p.keys():
    #    print(params)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # optimizer = torch.optim.SGD(
    # model.parameters(),
    # args.learning_rate,
    # momentum=args.momentum,
    # weight_decay=args.weight_decay)
    optimizer = MixOptimizer(model.parameters(), lr=args.learning_rate)
    data_loader = DataLoaderHDM05(args.data, pval, ptest, trainperc)
    train_queue = data_loader._train_generator
    valid_queue = data_loader._val_generator
    print(len(train_queue))
    print(len(valid_queue))
    print("Dataset loaded")

    architect = Architect(model.double().to(device), args)

    for epoch in range(args.epochs):
        #scheduler.step()
        #lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, args.learning_rate)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        n = model.alphas_reduce[0, :].shape[0]
        # for i in range(model.alphas_reduce.shape[0]):
        print(use_sparsemax_as_convex_layer(model.alphas_normal, n))
        print(use_sparsemax_as_convex_layer(model.alphas_reduce, n))
        print(
            torch.sum(use_sparsemax_as_convex_layer(model.alphas_normal, n),
                      dim=1))
        print(
            torch.sum(use_sparsemax_as_convex_layer(model.alphas_reduce, n),
                      dim=1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model,
                                     architect, criterion, optimizer,
                                     args.learning_rate)
        logging.info('train_acc %f', train_acc)

        print('Training acc: ' + str(train_acc.data.cpu()) + '% at epoch ' +
              str(epoch + 1) + '/' + str(args.epochs))

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        print('Val acc: ' + str(valid_acc.data.cpu()) + '% at epoch ' +
              str(epoch + 1) + '/' + str(args.epochs))

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer,
          lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).to(device)  # .cuda()
        target = Variable(target,
                          requires_grad=False).to(device)  # .cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search,
                                requires_grad=False).to(device)  # .cuda()
        target_search = Variable(target_search, requires_grad=False).to(
            device)  # .cuda(async=True)

        architect.step(input,
                       target,
                       input_search,
                       target_search,
                       args.learning_rate,
                       optimizer,
                       unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        n = model.alphas_reduce[0, :].shape[0]
        print(use_sparsemax_as_convex_layer(model.alphas_normal, n))
        print(use_sparsemax_as_convex_layer(model.alphas_reduce, n))
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                         top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        #input = Variable(input)# volatile=True)#.cuda()
        #target = Variable(target)# volatile=True)#.cuda(async=True)
        input = input.to(device)
        target = target.to(device)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg,
                         top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':

    params = [100]
    for perc in params:
        main(perc)
