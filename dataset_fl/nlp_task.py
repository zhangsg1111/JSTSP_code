import numpy as np
import torch
import pickle as pkl
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from datasets import load_dataset
from transformers import BertTokenizer


class MRPC_dataset(Dataset):
    def __init__(self, path, tokenizer, data_type='train', max_length=128):
        # data_type='train' or 'validation' or 'test'
        self.dataset = load_dataset(path)[data_type]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = [data['label'] for data in self.dataset]
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        label = data['label']
        len_sentence1_tokens = len(self.tokenizer.encode(sentence1, add_special_tokens=False))
        len_sentence2_tokens = len(self.tokenizer.encode(sentence2, add_special_tokens=False))
        seg = [0]*(len_sentence1_tokens+2) + [1]*(len_sentence1_tokens+1)
        if len(seg)<self.max_length:
            num_pad = self.max_length - len(seg)
            mask = [1]*len(seg)+[0]*num_pad
            seg = seg + [0]*num_pad
        else:
            mask = self.max_length
            seg = seg[0:self.max_length]
        input_ids = self.tokenizer.encode(sentence1+'[SEP]'+ sentence2, 
                         add_special_tokens=True, padding='max_length',
                         truncation=True, max_length=self.max_length, return_tensors='pt')
        return input_ids[0], torch.tensor(seg), torch.tensor(mask), torch.tensor(label)
    
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
    if dataset == "mrpc":
        transform_train = None
        transform_test = None
    return transform_train, transform_test

def get_nlp_dataset(dataset:str, data_dir, num_users, iid=True, alpha=0.5):
    transform_train, transform_test = get_transforms(dataset)
    if dataset == "mrpc":
        tokenizer = BertTokenizer.from_pretrained('/home/chengguoliang/project/SplitedLM/pretrained_weights/uncased_L-12_H-768_A-12') 
        train_dataset = MRPC_dataset(data_dir, tokenizer, data_type='train')
        test_dataset = MRPC_dataset(data_dir, tokenizer, data_type='test')
        num_classes = 2
    else:
        raise NameError("Can not find this dataset.")
        
    if iid:
        user_groups = get_iid_data(train_dataset, num_users)
    else:
        user_groups = get_noniid_data(train_dataset, num_users, alpha, num_classes)
        # user_groups = dataset_noniid_split(train_dataset, num_users, num_shards=400, num_shards_per_client=4)
    return train_dataset, test_dataset, user_groups, num_classes