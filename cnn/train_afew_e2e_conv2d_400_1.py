from torch.utils import data
from torch.autograd import Variable
from model_spd_v0_radar import Network
import torch.nn as nn
import random
import torch as th
import genotypes
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
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.autograd import Variable
from model_spd_v0_afew import Network
from optimizers import MixOptimizer
import os
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from model_spd_v0_radar import Network
import torch.nn as nn
import random
import torch as th
import genotypes
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
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.autograd import Variable
from model_spd_v0_afew_conv2d_400_1 import Network
#_400_100_50 import Network
import random
import torch as th
import genotypes
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from optimizers import MixOptimizer

parser = argparse.ArgumentParser("radar")
parser.add_argument('--data',
                    type=str,
                    default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay',
                    type=float,
                    default=3e-4,
                    help='weight decay')
parser.add_argument('--report_freq',
                    type=float,
                    default=50,
                    help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='num of training epochs')
parser.add_argument('--init_channels',
                    type=int,
                    default=36,
                    help='num of init channels')
parser.add_argument('--layers',
                    type=int,
                    default=20,
                    help='total number of layers')
parser.add_argument('--model_path',
                    type=str,
                    default='saved_models',
                    help='path to save the model')
parser.add_argument('--auxiliary',
                    action='store_true',
                    default=False,
                    help='use auxiliary tower')
parser.add_argument('--auxiliary_weight',
                    type=float,
                    default=0.4,
                    help='weight for auxiliary loss')
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
                    default=0.2,
                    help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch',
                    type=str,
                    default='DARTS',
                    help='which architecture to use')
parser.add_argument('--grad_clip',
                    type=float,
                    default=5,
                    help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

data_path = 'data/afew/'
batch_size = 16  # batch size

from read_afew_frames import *

data_path = 'data/afew/'
batch_size = 30  # batch size


class DatasetSPD(data.Dataset):
    def __init__(self, videos, labels):
        self._labels = labels
        self._videos = videos

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, item):
        #x = np.load(self._path + self._names[item])[None, :, :].real
        # [batch_size, in_channels, depth, height, width].
        if self._videos[item].shape[0] >= 32:
            #print("Non repeated")
            idx = np.random.choice(self._videos[item].shape[0],
                                   32,
                                   replace=False)
            #print(idx)
        else:
            #print("repeated")
            idx = np.random.choice(self._videos[item].shape[0], 32)
            #print(idx)
        x = self._videos[item][idx, :, :]
        x = np.moveaxis(x, -1, 1)
        x = th.from_numpy(x).double()
        y = int(self._labels[item])
        y = th.from_numpy(np.array(y)).long()
        return x, y


class DataLoaderAFEW:
    def __init__(self, data_path, batch_size):
        train_class = ReadAFEWTrain()
        val_class = ReadAFEWVal()
        train_labels, train_videos = train_class.get_dataset()
        inds = np.random.permutation(len(train_labels))
        train_labels = [train_labels[i] for i in inds]
        train_videos = [train_videos[i] for i in inds]
        val_labels, val_videos = val_class.get_dataset()
        self._train_generator = data.DataLoader(DatasetSPD(
            train_videos, train_labels),
                                                batch_size=batch_size,
                                                shuffle='True')
        self._test_generator = data.DataLoader(DatasetSPD(
            val_videos, val_labels),
                                               batch_size=batch_size,
                                               shuffle='False')


def main():
    num_classes = 7
    # if not torch.cuda.is_available():
    #  logging.info('no gpu device available')
    #  sys.exit(1)
    args.layers = 2
    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    # torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)
    args.save = 'search-{}-{}'.format(args.save,
                                      time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    #data_path = 'data/afew/'  # data path
    #pval = 0.25  # validation percentage
    #ptest = 0.25  # test percentage
    #batch_size = 30  # batch size
    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()
    genotype = eval("genotypes.%s" % args.arch)
    #writer = SummaryWriter()
    model = Network(1, 400, 50, num_classes, 2, genotype)
    model = model.double()
    p = model.state_dict()
    for params in p.keys():
        print(params)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    optimizer = MixOptimizer(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.SGD(
    #  model.parameters(),
    #  args.learning_rate,
    #  momentum=args.momentum,
    #  weight_decay=args.weight_decay)
    data_loader = DataLoaderAFEW(data_path, batch_size)
    train_queue = data_loader._train_generator
    #valid_queue = data_loader._val_generator
    test_queue = data_loader._test_generator
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    args.epochs = 200
    for epoch in range(args.epochs):
        #scheduler.step()
        #logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        #writer.add_scalar('Loss/train', objs.avg, step)
        #writer.add_scalar('Acc/train', train_acc, epoch)
        #valid_acc, valid_obj = infer(valid_queue, model, criterion)
        #logging.info('valid_acc %f', valid_acc)
        #writer.add_scalar('Acc/valid', valid_acc, epoch)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        test_acc, test_obj = infer(test_queue, model, criterion)
        print("Test acc", test_acc)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input)  #.cuda()
        target = Variable(target)  #.cuda(async=True)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        #if args.auxiliary:
        #  loss_aux = criterion(logits_aux, target)
        #  loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top2.update(prec5.data, n)

        if step % 1 == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                         top2.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True)  #.cuda()
        target = Variable(target, volatile=True)  #.cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top2.update(prec5.data, n)

        if step % 1 == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg,
                         top2.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
