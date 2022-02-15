import os
import sys
import time
import glob
import numpy as np
import torch
import time
from gumbel_softmax import gumbel_softmax
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
from model_spd_v0_radar_search_sparsemax import Network
from architect import Architect
import matplotlib.pyplot as plt
import random
from analyze import Analyzer
import torch as th
from optimizers import MixOptimizer
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F

parser = argparse.ArgumentParser("afew")
parser.add_argument('--data',
                    type=str,
                    default='data/hdm05/',
                    help='location of the data corpus')
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
                    default=20,
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
parser.add_argument('--seed', type=int, default=2, help='random seed')
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
                    default=False,
                    help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate',
                    type=float,
                    default=3e-4,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay',
                    type=float,
                    default=1e-3,
                    help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
data_path = 'data/radar/'  # data path
pval = 0.25  # validation percentage
ptest = 0.25  # test percentagei
batch_size = 32  # batch size
"""def use_softmax_as_convex_layer(x, n):
    x_ = cp.Parameter(n)
    y_ = cp.Variable(n)

    #define the dual objective of the softmax in dual form
    objective = cp.Minimize(-cp.sum(cp.multiply(x_, y_)) - cp.sum(cp.entr(y_)))
    constraint = [cp.sum(y_) == 1.]

    #optimization problem
    optimization_problem = cp.Problem(objective, constraint)
    layer = CvxpyLayer(optimization_problem, parameters=[x_], variables=[y_])
    y, = layer(x)
    return y"""


def use_sparsemax_as_convex_layer(x, n):
    x_ = cp.Parameter(n)
    y_ = cp.Variable(n)
    obj = cp.sum_squares(x_ - y_)
    cons = [cp.sum(y_) == 1, 0. <= y_, y_ <= 1.]
    prob = cp.Problem(cp.Minimize(obj), cons)
    layer = CvxpyLayer(prob, [x_], [y_])
    y, = layer(x)
    return y


class DatasetRadar(data.Dataset):

    def __init__(self, path, names):
        self._path = path
        self._names = names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        x = np.load(self._path + self._names[item])
        x = np.concatenate((x.real[:, None], x.imag[:, None]), axis=1).T
        x = th.from_numpy(x)
        y = int(self._names[item].split('.')[0].split('_')[-1])
        y = th.from_numpy(np.array(y))
        return x.double(), y.long()


class DataLoaderRadar:

    def __init__(self, data_path, pval, batch_size):
        for filenames in os.walk(data_path):
            names = sorted(filenames[2])
        random.Random(8).shuffle(names)
        N_val = int(pval * len(names))
        N_test = int(ptest * len(names))
        N_train = len(names) - N_test - N_val
        train_set = DatasetRadar(
            data_path, names[N_val + N_test:int(N_train) + N_test + N_val])
        test_set = DatasetRadar(data_path, names[:N_test])
        val_set = DatasetRadar(data_path, names[N_test:N_test + N_val])
        self._train_generator = data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle='True')
        self._test_generator = data.DataLoader(test_set,
                                               batch_size=batch_size,
                                               shuffle='False')
        self._val_generator = data.DataLoader(val_set,
                                              batch_size=batch_size,
                                              shuffle='False')


num_classes = 3


def main():
    #if not torch.cuda.is_available():
    #  logging.info('no gpu device available')
    #  sys.exit(1)
    args.layers = 2
    temp_start = 1
    temp_min = 0.5
    decay = 0.99
    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    #torch.cuda.manual_seed(args.seed)
    #logging.info('gpu device = %d' % args.gpu)
    #logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    #criterion = criterion.cuda()
    model = Network(1, 20, 10, num_classes, 2, criterion)
    model = model.double()
    p = model.state_dict()
    for params in p.keys():
        print(params)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    optimizer = MixOptimizer(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.SGD(
    #    model.parameters(),
    #    args.learning_rate,
    #    momentum=args.momentum,
    #    weight_decay=args.weight_decay)
    data_loader = DataLoaderRadar(data_path, pval, batch_size)
    train_queue = data_loader._train_generator
    valid_queue = data_loader._val_generator
    print(len(train_queue))
    print(len(valid_queue))
    print("Dataset loaded")
    analyser = Analyzer(args, model)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #      optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    #writer = SummaryWriter()
    architect = Architect(model.double(), args)
    #args.epochs=20
    #eigs_epochs=[]
    for epoch in range(args.epochs):
        #scheduler.step()
        time_start = time.time()
        lr = args.learning_rate
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        n = model.alphas_reduce[0, :].shape[0]
        #for i in range(model.alphas_reduce.shape[0]):
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
                                     architect, criterion, optimizer, analyser,
                                     lr, temp_start, temp_min, decay)
        logging.info('train_acc %f', train_acc)
        #plt.plot(eig_vals)
        #eigs_epochs.append(eig_vals)
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        time_end = time.time()
        print("Time for epoch", time_end - time_start)
    #flat_list_eigs= [item for sublist in eig_vals for item in sublist]
    #plt.plot(flat_list_eigs)


def train(train_queue, valid_queue, model, architect, criterion, optimizer,
          analyser, lr, temp_start, temp_min, decay):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    temp = temp_start
    eig_vals = []
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False)  # .cuda()
        target = Variable(target, requires_grad=False)  # .cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False)  # .cuda()
        target_search = Variable(target_search,
                                 requires_grad=False)  # .cuda(async=True)

        architect.step(input,
                       target,
                       input_search,
                       target_search,
                       lr,
                       optimizer,
                       unrolled=args.unrolled)
        #H = analyser.compute_Hw(input, target, input_search, target_search,
        #                        lr, optimizer,temp, False)
        #eigs=analyser.compute_eigenvalues()
        #eig_vals.append(np.max(eigs))
        #print("Max eig value",np.max(eigs))
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        n = model.alphas_reduce[0, :].shape[0]
        print(use_sparsemax_as_convex_layer(model.alphas_normal, n))
        print(use_sparsemax_as_convex_layer(model.alphas_reduce, n))
        temp = min(temp * decay, temp_min)
        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec2.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                         top5.avg)
        #writer.add_scalar('Loss/train',objs.avg, step)
        #writer.add_scalar('Acc/train', top1.avg, step)
        #break
    return top1.avg, objs.avg  #,eig_vals


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        #input = Variable(input)# volatile=True)#.cuda()
        #target = Variable(target)# volatile=True)#.cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec2.data, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg,
                         top5.avg)
        #writer.add_scalar('Loss/val', objs.avg, step)
        #break
        #writer.add_scalar('Acc/val', top1.avg, step)
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
