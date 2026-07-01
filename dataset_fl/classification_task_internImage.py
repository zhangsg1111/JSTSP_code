import numpy as np
import torch
import pickle as pkl
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import datasets
    
class ClsDataset(Dataset):
    def __init__(self, x, y, transform, classes=None):
        self.x = x
        self.y = y
        self.targets = self.y
        self.classes = classes
        self.transform = transform
        
    def __len__(self):
        num_x = len(self.x)
        num_y = len(self.y)
        assert num_x==num_y
        return num_x
    
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.x[idx]), self.y[idx]
        else:
            return self.x[idx], self.y[idx].item()

class TinyImagenet_Dataset(Dataset):
    def __init__(self, data_dict, transform, classes=200):
        self.data_dict = data_dict
        self.classes = classes
        self.targets = data_dict['label']
        self.transform = transform
        
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        image = self.data_dict[idx]['image'].convert('RGB')
        label = self.data_dict[idx]['label']
        if self.transform is not None:
            return self.transform(image), label
        else:
            return image, label
    
# Independent Identically Distributed(IID)
def get_iid_data(dataset, num_users): # get IID-distribution data 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def get_noniid_data(dataset, num_users, alpha, num_classes):# get non-IID-distribution data 
    np.random.seed(0)
    """
    dataset: training set of CIFAR
    """
    dict_users = {}
    min_size = 0
    labels = np.array(dataset.targets)
    num_items = int(len(dataset)/num_users)
    while min_size < 10:
        idx_groups = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            # Balance
            proportions = np.array(
                [p*(len(user_idx) < num_items) for p, user_idx in zip(proportions, idx_groups)]
                )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_groups = [user_idx + idx.tolist() for user_idx, idx in zip(idx_groups, np.split(idx_k, proportions))]
            min_size = min([len(user_idx) for user_idx in idx_groups])
    for i in range(num_users):
        np.random.shuffle(idx_groups[i])
        dict_users[i] = idx_groups[i]
    return dict_users

def get_transforms(dataset):
    if dataset == "CIFAR10":
        transform_train = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomCrop(384, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == "CIFAR100":
        transform_train = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomCrop(384, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                  (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset=="MNIST" or dataset=="FMNIST":
        transform_train = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif dataset=='Tiny_Imagenet':
        print(dataset)
        transform_train = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomCrop(384, padding=32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transform_train, transform_test

def get_classification_dataset_internImage(dataset:str, data_dir, num_users, iid=True, alpha=0.5):
    transform_train, transform_test = get_transforms(dataset)
    if dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=transform_test)
        num_classes = 10
        fig_size = (3, 384, 384)
        
    elif dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True,
                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(data_dir, train=False, download=True,
                                    transform=transform_test)
        num_classes = 100
        fig_size = (3, 384, 384)
        
    elif dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                     transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                    transform=transform_test)
        num_classes = 10
        fig_size = (1, 384, 384)
        
    elif dataset == "FMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True,
                                     transform=transform_train)
        test_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True,
                                    transform=transform_test)
        num_classes = 10
        fig_size = (1, 384, 384)
        
    elif dataset == "Tiny_Imagenet":
        data_dict = datasets.load_dataset(data_dir)
        train_dataset = TinyImagenet_Dataset(data_dict['train'], transform_train)
        test_dataset = TinyImagenet_Dataset(data_dict['valid'], transform_test)
        num_classes = 200
        fig_size = (3, 384, 384)
    else:
        raise NameError("Can not find this dataset.")
        
    if iid:
        user_groups = get_iid_data(train_dataset, num_users)
    else:
        user_groups = get_noniid_data(train_dataset, num_users, alpha, num_classes)
        # user_groups = dataset_noniid_split(train_dataset, num_users, num_shards=400, num_shards_per_client=4)
    return train_dataset, test_dataset, user_groups, num_classes, fig_size