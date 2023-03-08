import torch
import pickle
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class cifar100_dataset(Dataset):
    def __init__(self, root_path, train=True, transform = None):
        super().__init__()
        self.transform = transform
        if train:
            self.data_dict = unpickle(os.path.join(root_path, 'cifar-100-python', 'train'))
        else:
            self.data_dict = unpickle(os.path.join(root_path, 'cifar-100-python', 'test'))
        self.data = self.data_dict['data'].reshape(-1,3,32,32).transpose((0, 2, 3, 1))          # [N, H, W, C]
        self.labels = self.data_dict['fine_labels']

    def __getitem__(self, index):
        temp_data = self.data[index]
        temp_label = self.labels[index]
        img = Image.fromarray(temp_data)
        if self.transform is not None:
            img = self.transform(img)
        return img, temp_label

    def __len__(self):
        return len(self.data)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


if __name__=='__main__':
    root_path = './../data/cifar-100-python'
    cifar100_name = 'train'
    data = unpickle(os.path.join(root_path, cifar100_name))
    train_data = data["data"]
    train_label = data["fine_labels"]


    trainset = datasets.CIFAR100(root='../data', train=True, download=True,transform=transforms.Compose([
                transforms.ToTensor()]))
    own_set = cifar100_dataset(root_path='./../data/', train=True, transform=transforms.Compose([transforms.ToTensor()]))
    own_trainloader = torch.utils.data.DataLoader(own_set, batch_size=4, shuffle= False, num_workers= 4, pin_memory= True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = False,num_workers = 4, pin_memory = True)
