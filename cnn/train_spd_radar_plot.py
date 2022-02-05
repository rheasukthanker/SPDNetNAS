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
from model_spd_v0_radar import Network
import random
import pickle
import torch as th
import genotypes
from torch.utils.tensorboard import SummaryWriter
from optimizers import MixOptimizer

parser = argparse.ArgumentParser("radar")
parser.add_argument('--data',
                    type=str,
                    default='data/radar/',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.025,
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
                    default='v0_radar_gen1',
                    help='which architecture to use')  # CHANGE ACCORDINGLY
parser.add_argument('--fulltraining',
                    action='store_true',
                    default=False,
                    help='Whether we train with full trainset (no validation)')
parser.add_argument('--grad_clip',
                    type=float,
                    default=5,
                    help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}-fulltrain-{}-epoch-{}'.format(args.arch,
                                                      time.strftime("m%d-M%S"),
                                                      args.fulltraining,
                                                      args.epochs)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

RADAR_CLASSES = 3
pval = 0.25  # validation percentage
ptest = 0.25  # test percentage


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
    def __init__(self,
                 data_path,
                 pval,
                 ptest,
                 trainperc=100,
                 fulltraining=False):
        for filenames in os.walk(data_path):
            names = sorted(filenames[2])
        print(len(names))
        random.Random(0).shuffle(names)
        N_val = int(pval * len(names))
        N_test = int(ptest * len(names))

        if fulltraining == True:
            N_train = int(trainperc * 0.01 * (len(names) - N_test))
        else:
            N_train = int(trainperc * 0.01 * (len(names) - N_test - N_val))

        test_set = DatasetRadar(data_path, names[:N_test])

        if fulltraining == True:
            train_set = DatasetRadar(data_path, names[N_test:N_test + N_train])
            self._train_generator = data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle='True')
            self._test_generator = data.DataLoader(
                test_set, batch_size=args.batch_size,
                shuffle='False')  # NOT USING IT FOR SEARCH
        else:
            val_set = DatasetRadar(data_path, names[N_test:N_test + N_val])
            train_set = DatasetRadar(
                data_path, names[N_val + N_test:N_val + N_test + N_train]
            )  # trainsize = Nb of training samples, default is 782 (37.5%)
            self._train_generator = data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle='True')
            self._val_generator = data.DataLoader(val_set,
                                                  batch_size=args.batch_size,
                                                  shuffle='False')
            self._test_generator = data.DataLoader(
                test_set, batch_size=args.batch_size,
                shuffle='False')  # NOT USING IT FOR SEARCH


def main(trainperc):
    num_classes = 3
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
    logging.info("args = %s", args)
    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(1, 20, 10, num_classes, 2, genotype)
    #model.load_state_dict(torch.load("C:/Users/41783/Documents/darts/cnn/search-eval-EXP-20200529-103205-20200529-103205/weights.pt"))
    model = model.double()
    p = model.state_dict()
    for params in p.keys():
        print(params)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    optimizer = MixOptimizer(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(
    # model.parameters(),
    #  args.learning_rate,
    #  momentum=args.momentum,
    #  weight_decay=args.weight_decay)
    data_loader = DataLoaderRadar(args.data, pval, ptest, trainperc,
                                  args.fulltraining)
    train_queue = data_loader._train_generator
    if args.fulltraining == False:
        valid_queue = data_loader._val_generator
        print(len(valid_queue))
    test_queue = data_loader._test_generator
    print(len(train_queue))
    print(len(test_queue))

    ## WRITER
    #comment = f' Radar final training arch_{args.arch} - (lr = {args.learning_rate})- Percentage of tot. number of Training Samples = {trainperc} - full training = {args.fulltraining} - epoch = {args.epochs} '  # AÑADIR ARCH
    writer = SummaryWriter()
    test_accs = []
    for epoch in range(args.epochs):
        # scheduler.step()
        logging.info('epoch %d lr %e', epoch, args.learning_rate)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if epoch == 0:
            starttime = time.time()
            train_acc, train_obj = train(train_queue,
                                         model,
                                         criterion,
                                         optimizer,
                                         init=True)
            if args.fulltraining == False:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                writer.add_scalar('Validation_accuracy_w.r.t_time', valid_acc,
                                  0)
            writer.add_scalar('Training_accuracy_w.r.t_time', train_acc, 0)
            if args.fulltraining == False:
                print('Initial val. acc: ' +
                      str(valid_acc.data.cpu().numpy()) +
                      '% and train. acc. ' + str(train_acc.data.cpu().numpy()))
            else:
                print('Train. acc. ' + str(train_acc.data.cpu().numpy()))

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        ## WRITE TRAINING ACCURACY W.R.T TIME AND EPOCH
        t = round((time.time() - starttime) / 60, 1)
        writer.add_scalar('Training_accuracy_w.r.t_time', train_acc, t)
        writer.add_scalar('Training_accuracy_w.r.t_epoch', train_acc, epoch)
        writer.add_scalar('Training_Loss_w.r.t_epoch', train_obj, epoch)
        print('Training acc: ' + str(train_acc.data.cpu().numpy()) +
              ' % at epoch ' + str(epoch + 1) + '/' + str(args.epochs))

        if args.fulltraining == False:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            ## WRITE VALIDATION ACCURACY W.R.T TIME AND EPOCH
            t = round((time.time() - starttime) / 60, 1)
            writer.add_scalar('Validation_accuracy_w.r.t_time', valid_acc, t)
            writer.add_scalar('Validation_accuracy_w.r.t_epoch', valid_acc,
                              epoch)
            writer.add_scalar('Validation_Loss_w.r.t_epoch', valid_obj, epoch)
            print('Val acc: ' + str(valid_acc.data.cpu().numpy()) +
                  '% at epoch ' + str(epoch + 1) + '/' + str(args.epochs))

        utils.save(model, os.path.join(args.save, 'weights.pt'))

        test_acc, test_obj = infer(test_queue, model, criterion)
        test_accs.append(test_acc)
        logging.info('test_acc %f', test_acc)
        writer.add_scalar('Test_accuracy', test_acc, len(train_queue.dataset))
        print('Test acc: ' + str(test_acc.data.cpu().numpy()))
    writer.close()
    with open("test_acc_radar_100_2.pkl", "wb") as f:
        pickle.dump(test_accs, f)


def train(train_queue, model, criterion, optimizer, init=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input)  # .cuda()
        n = input.size(0)
        target = Variable(target)  # .cuda(async=True)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        if init == True:
            prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top2.update(prec2.data, n)
            return top1.avg, objs.avg

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))

        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top2.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg,
                         top2.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True)  # .cuda()
        target = Variable(target, volatile=True)  # .cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top2.update(prec5.data, n)

        if step % args.report_freq == 0 and valid_queue == 'valid_queue':
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg,
                         top2.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    '''Here goes the percentage of training data you want your model to be trained with'
    We could do something like this -> params = [100,80,33,10]'''
    params = [100]
    for perc in params:
        main(perc)
