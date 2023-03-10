import time
import argparse
import numpy as np
import torch
from data import get_dataset
from model import get_model
import torchvision.transforms as transforms
import torch.nn as nn
from util import AverageMeter, accuracy, set_seed
import os
from progress.bar import Bar as Bar
from torchvision.models import resnet18
from tensorboardX import SummaryWriter
import logging


def train(model, train_loader, criterion, optimizer, logger):
    epoch_start_time = time.time()
    model.train()
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    train_bar = Bar('training', max=len(train_loader))
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_losses.update(loss.item(), data.size(0))
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        train_top1.update(prec1, data.size(0))
        train_top5.update(prec5, data.size(0))
        # print(loss)
        loss.backward()
        optimizer.step()
        train_bar.suffix = '({batch}/{size} | time: {time:.2f} | Loss: {loss} | top1: {top1} | top5: {top5})'.format(
            batch=batch_id + 1, size=len(train_loader), time=time.time() - epoch_start_time, loss=train_losses.avg,
            top1=train_top1.avg, top5=train_top5.avg)
        train_bar.next()
    train_bar.finish()
    return train_losses, train_top1, train_top5


def test(model, test_loader, criterion):
    epoch_start_time = time.time()
    model.eval()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    test_bar = Bar('testing', max=len(test_loader))
    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            prec1, prec5 = accuracy(output, target, (1, 5))
            test_top1.update(prec1, data.size(0))
            test_top5.update(prec5, data.size(0))
            test_bar.suffix = '({batch}/{size} | time: {time:.2f} | top1: {top1} | top5: {top5})'.format(
                batch=batch_id + 1, size=len(test_loader), time=time.time() - epoch_start_time, top1=test_top1.avg,
                top5=test_top5.avg)
            test_bar.next()
    test_bar.finish()
    return test_top1, test_top5


def main():
    parser = argparse.ArgumentParser(description='a simple example')
    # model and data
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--datasets', default='CIFAR100')
    parser.add_argument('--data_root', default='./../data/')
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)

    # optimizer
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cos'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225])
    parser.add_argument('--gamma', type=float, default=0.1)

    # save logs
    parser.add_argument('--save_dir', default='./../backup_check/task_cifar100/')
    parser.add_argument('--name', default='test')

    # other setup
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.random_seed)

    # model and data
    train_set, test_set = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    print('train loader length: {}'.format(len(train_loader.dataset)))
    print('test loader length: {}'.format(len(test_loader.dataset)))

    # ori_model = resnet18().cuda()
    model = get_model(args)
    model = torch.nn.DataParallel(model).cuda()

    # optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    elif args.lr_schedule == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=4e-4)

    best_acc = 0
    best_epoch = 0

    # save files
    save_dir = os.path.join(args.save_dir, args.name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    logger = SummaryWriter(log_dir=save_dir)
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename=os.path.join(save_dir, 'test.log'),
                        filemode='a')
    logging.info(args)

    for epoch in range(200):
        print('epoch:[{}/200] lr: {}'.format(epoch + 1, scheduler.get_last_lr()[0]))
        train_losses, train_top1, train_top5 = train(model, train_loader, criterion, optimizer, logger)
        logger.add_scalar('train_loss', train_losses.avg, epoch)
        logger.add_scalar('train_top1', train_top1.avg, epoch)
        logger.add_scalar('train_top5', train_top5.avg, epoch)
        scheduler.step()
        test_top1, test_top5 = test(model, test_loader, criterion)
        logger.add_scalar('test_top1', test_top1.avg, epoch)
        logger.add_scalar('test_top5', test_top5.avg, epoch)
        logging.info(
            'epoch:{}/{} temp lr : {}  train loss : {} train_top1 : {} train_top5 : {}   test top1 : {} test top5 : {} '.format(
                epoch + 1, 200,
                scheduler.get_last_lr()[0],
                train_losses.avg,
                train_top1.avg,
                train_top5.avg,
                test_top1.avg,
                test_top5.avg))
        if best_acc < test_top1.avg:
            best_acc = test_top1.avg
            best_epoch = epoch + 1
    print('Best acc: {} in epoch: {}'.format(best_acc, best_epoch))

    #     print(
    #         'epoch:{}/{} temp lr:{}  train loss:{} train_top1:{} train_top5:{}   test top1:{} test top5:{} '.format(
    #             epoch + 1, 200,
    #             scheduler.get_last_lr()[0],
    #             train_loss.avg,
    #             train_top1.avg,
    #             train_top5.avg,
    #             test_top1.avg,
    #             test_top5.avg))
    # pass


if __name__ == '__main__':
    main()
