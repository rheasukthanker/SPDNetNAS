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
from model_spdnet_v1_radar_search import Network
from architect import Architect
import random
import torch as th
from torch.utils.tensorboard import SummaryWriter

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
ptest = 0.25  # test percentage
batch_size = 30  # batch size


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
        random.Random().shuffle(names)
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
    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    data_loader = DataLoaderRadar(data_path, pval, batch_size)
    train_queue = data_loader._train_generator
    valid_queue = data_loader._val_generator
    print(len(train_queue))
    print(len(valid_queue))
    print("Dataset loaded")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    writer = SummaryWriter()
    architect = Architect(model.double(), args)
    args.epochs = 20
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model,
                                     architect, criterion, optimizer, lr,
                                     writer)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, writer)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,
          writer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

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

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))
        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec2.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                         top5.avg)
        writer.add_scalar('Loss/train', objs.avg, step)
        writer.add_scalar('Acc/train', top1.avg, step)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, writer):
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
        writer.add_scalar('Loss/val', objs.avg, step)
        writer.add_scalar('Acc/val', top1.avg, step)
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
