import torch
import pandas
from model import get_model
from data import get_dataset
import argparse
from openpyxl import Workbook
from progress.bar import Bar as Bar
import os


def get_position(output, target):
    batch_size = output.size(0)
    _, pred = output.topk(1, 1, True, True)  # [b 1]
    pred = pred.t()  # [1,b]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred)).float()

    return correct


def main():
    parser = argparse.ArgumentParser(description='a simple example')
    parser.add_argument('--check_num', default='200')
    parser.add_argument('--check_name', default='test')
    # model and data
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--datasets', default='CIFAR100')
    parser.add_argument('--data_root', default='./../data/')
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=6, type=int)
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
    parser.add_argument('--save_path', action='store_true')

    # other setup
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--random_seed', default=0, type=int)

    args = parser.parse_args()
    print(args)
    check_points = os.path.join(args.save_dir, args.check_name, 'paths', "save_net_resnet18_" + args.check_num + ".pt")
    # check_points = "/home/DiskB/rqding/backup_check/task_cifar100/save_seed0_cos/paths/save_net_resnet18_"+args.check_num+".pt"
    load_check = torch.load(check_points)
    model = get_model(args)

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(load_check["state_dict"])

    train_set, test_set = get_dataset(args, get_index=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    print('train loader length: {}'.format(len(train_loader.dataset)))
    print('test loader length: {}'.format(len(test_loader.dataset)))
    wb = Workbook()  # 创建工作簿
    ws = wb.active
    ws.append(['index', 'label', 'predict', 'origin'])
    train_bar = Bar('training', max=len(test_loader))

    for batch_id, (data, label, index) in enumerate(test_loader):
        model.eval()
        with torch.no_grad():
            data, label = data.cuda(), label.cuda()
            out = model(data)  # [b, c]
            soft_out = torch.nn.functional.softmax(out, dim=1)
            _, pred = out.topk(1, 1, True, True)  # [b 1]

            # pred = pred.t()  # [1,b]
            batch_size = out.size(0)
            for write_i in range(batch_size):
                # print(index[write_i])
                # print(int(label[write_i]))
                # print(int(pred[write_i, 0]))
                # print('----------------')
                temp_predict = int(pred[write_i, 0])
                ws.append([int(index[write_i]), int(label[write_i]), temp_predict, int(soft_out[write_i, temp_predict])])
            train_bar.next()
    train_bar.finish()
    save_files = os.path.join('./../backup_check/save_results/', 'save_result_{}_{}.xlsx'.format(args.check_name, args.check_num))
    wb.save(save_files)

    pass


if __name__ == "__main__":
    main()
