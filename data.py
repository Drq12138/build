import torch
import pickle
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


class cifar100_dataset(Dataset):
    def __init__(self, root_path, train=True, transform=None, get_index=False):
        super().__init__()
        self.transform = transform
        self.get_index = get_index
        if train:
            self.data_dict = unpickle(os.path.join(root_path, 'cifar-100-python', 'train'))
        else:
            self.data_dict = unpickle(os.path.join(root_path, 'cifar-100-python', 'test'))
        self.data = self.data_dict['data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))  # [N, H, W, C]
        self.labels = self.data_dict['fine_labels']

    def __getitem__(self, index):
        temp_data = self.data[index]
        temp_label = self.labels[index]
        img = Image.fromarray(temp_data)
        if self.transform is not None:
            img = self.transform(img)
        if self.get_index:
            return img, temp_label, index
        else:
            return img, temp_label

    def __len__(self):
        return len(self.data)


def unpickle(file: object) -> object:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def get_dataset(args, get_index=False):
    if args.datasets == 'CIFAR100':
        train_transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        train_set = cifar100_dataset(root_path=args.data_root, train=True, transform=train_transform,
                                     get_index=get_index)
        test_set = cifar100_dataset(root_path=args.data_root, train=False, transform=test_transform,
                                    get_index=get_index)
        return train_set, test_set


if __name__ == '__main__':
    root_path = './../data/'
    save_dir = './../data/origin_data/'
    # train_path = os.path.join(root_path, 'cifar-100-python', 'train')
    # test_path = os.path.join(root_path, 'cifar-100-python', 'test')
    # train_unzip = unpickle(train_path)
    # train_data = train_unzip["data"]
    # train_label = train_unzip["fine_labels"]
    # test_unzip = unpickle(test_path)
    # test_data = test_unzip["data"]
    # test_label = test_unzip["fine_labels"]
    # data_name = "CIFAR100"
    # origin_root = os.path.join(save_dir, data_name)
    # origin_train = os.path.join(origin_root, 'train')
    # origin_test = os.path.join(origin_root, 'test')
    # total_count = 0
    # for i in range(len(test_data)):
    #     temp = test_data[i, :].reshape(3, 32, 32).transpose((1, 2, 0))
    #     img = Image.fromarray(temp)
    #     temp_label = test_label[i]
    #     if not os.path.isdir(os.path.join(origin_test, str(temp_label))):
    #         os.mkdir(os.path.join(origin_test, str(temp_label)))
    #     img.save(os.path.join(origin_test, str(temp_label), str(total_count)+ '.jpg'))
    #     total_count += 1
    #
    # img = Image.fromarray(temp)
    # img.save(os.path.join(origin_train, 'test.jpg'))

    trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()]))
    own_set = cifar100_dataset(root_path='./../data/', train=True,
                               transform=transforms.Compose([transforms.ToTensor()]))
    own_trainloader = torch.utils.data.DataLoader(own_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
