import time

import torch
from data import cifar100_dataset
from model import get_resnet18
import torchvision.transforms as transforms
import torch.nn as nn
from util import AverageMeter, accuracy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def test(model, test_loader, criterion):
    model.eval()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            prec1, prec5 = accuracy(output, target, (1, 5))
            test_top1.update(prec1, data.size(0))
            test_top5.update(prec5, data.size(0))
    return test_top1, test_top5


def main():
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    train_set = cifar100_dataset(root_path='./../data/', train=True,
                                 transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(32, 4),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))
    test_set = cifar100_dataset(root_path='./../data/', train=False,
                                transform=transforms.Compose([transforms.ToTensor(),normalize]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    print('train loader length: {}'.format(len(train_loader.dataset)))
    print('test loader length: {}'.format(len(test_loader.dataset)))

    model = get_resnet18(100).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
    for epoch in range(200):
        epoch_time = time.time()
        model.train()
        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss.update(loss.item(), data.size(0))
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            train_top1.update(prec1, data.size(0))
            train_top5.update(prec5, data.size(0))
            # print(loss)
            loss.backward()
            optimizer.step()
        train_time = time.time() - epoch_time
        test_top1, test_top5 = test(model, test_loader, criterion)
        print('epoch:{}/{} train loss:{} train_top1:{} train_top5:{}   test top1:{} test top5:{} trian time:{}'.format(epoch + 1, 200,
                                                                                                         train_loss.avg,
                                                                                                         train_top1.avg,
                                                                                                         train_top5.avg,
                                                                                                         test_top1.avg,
                                                                                                         test_top5.avg,
                                                                                                         train_time))
    pass


if __name__ == '__main__':
    main()
