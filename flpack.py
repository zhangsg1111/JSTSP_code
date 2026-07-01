import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from utils import binary_search

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs, indi_err=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.indi_err = indi_err
        self.targets = [self.dataset[idx][1] for idx in self.idxs]
        self.num_classes = len(set(dataset.targets))
                         
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.indi_err is not None: 
            if self.indi_err[item] == True:
                label = (label+random.randint(1,9))%10
        return image.clone().detach(), torch.tensor(label)


class GradAggregator:
    def __init__(self, glo_model, device):
        self.device = device
        self.glo_model = glo_model

    @torch.no_grad()
    def agg(self, grad, n0):
        total_n = self.n + n0
        for i, g in enumerate(grad):
            self.average_grad[i].mul_(self.n/total_n)
            g.mul_(n0 / total_n)
            self.average_grad[i].add_(g)
        self.n = total_n
        
    @torch.no_grad()
    def glo_model_update(self):
        for j, global_p in enumerate(self.glo_model.parameters()):
            global_p.data += self.average_grad[j]
            
    @torch.no_grad()
    def zero_n(self):
        self.n = 0
        self.average_grad = []
        for para in self.glo_model.parameters():
            average_grad_zero = torch.zeros_like(para).to(self.device)
            self.average_grad.append(average_grad_zero)
    
    
class Aggregator:                                                              
    def __init__(self, glo_model, device):
        self.device = device
        self.glo_model = glo_model

    @torch.no_grad()    
    def agg(self, local_model, n0):
        total_n = self.n + n0       
        for i, local_p in enumerate(local_model.parameters()):
            self.average_model[i].mul_(self.n / total_n)
            local_p.mul_(n0 / total_n)   
            self.average_model[i].add_(local_p)
        self.n = total_n  
    
    def glo_model_update(self):
        for j, global_p in enumerate(self.glo_model.parameters()):
            global_p.data = self.average_model[j]
    
    def zero_n(self):
        self.n = 0
        self.average_model = []
        for para in self.glo_model.parameters():
            average_model_zero = torch.zeros_like(para).to(self.device)
            self.average_model.append(average_model_zero)
            
            
class EWGradAggregator: # element-wise GradAggregator
    def __init__(self, glo_model, device):
        self.device = device
        self.glo_model = glo_model

    @torch.no_grad()
    def agg(self, grad, n0, width_alpha_list):
        for i, (g, width_alpha) in enumerate(zip(grad, width_alpha_list)):
            num_units = g.size(0)
            num_units_f = int(num_units*width_alpha)
            n0_g = torch.ones_like(self.n[i], device=self.n[i].device)*n0
            n0_g[num_units_f:] = 0
            total_n_g = self.n[i] + n0_g
            zeros_mask = (total_n_g==0)
            total_n_g[zeros_mask] = 1e-8
            self.average_grad[i].mul_(self.n[i]/total_n_g)
            g.mul_(n0_g / total_n_g)
            self.average_grad[i].add_(g)
            total_n_g[zeros_mask] = 0
            self.n[i] = total_n_g

    @torch.no_grad()
    def glo_model_update(self):
        for j, global_p in enumerate(self.glo_model.parameters()):
            global_p.data += self.average_grad[j]
            
    @torch.no_grad()
    def zero_n(self):
        self.n = []
        self.average_grad = []
        for para in self.glo_model.parameters():
            average_grad_zero = torch.zeros_like(para).to(self.device)
            self.average_grad.append(average_grad_zero)
            n = torch.zeros((para.size(0)), device=self.device)
            n = n.view([para.size(0)]+[1]*(len(para.size())-1))
            self.n.append(n)
            

class Server:
    def __init__(self, num_total_clients, test_set, glo_model, args, device="cuda"):
        self.device = device
        self.num_total_clients = num_total_clients
        self.glo_model = glo_model.to(self.device)
        self.testloader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        # self.aggregator = Aggregator(self.glo_model, self.device)
        self.aggregator = GradAggregator(self.glo_model, self.device)
        self.ew_aggregator = EWGradAggregator(self.glo_model, self.device)
        self.aggregator.zero_n()
        self.ew_aggregator.zero_n()
    
    def choice_clients(self, c):
        choice_num = max(1, int(c*self.num_total_clients))
        choice_idx = np.random.choice([i for i in range(self.num_total_clients)], choice_num, replace=False)
        choice_idx = np.sort(choice_idx)
        return choice_idx
    
    @torch.no_grad()
    def eval_test(self, model):
        total_num_data = len(self.testloader.dataset)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= total_num_data
        print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, total_num_data,
            100. * correct / total_num_data))
        return 100. *correct/total_num_data, test_loss
    
    @torch.no_grad()
    def get_grad(self, model, model0):
        grads = []
        for p1, p2 in zip(model.parameters(), model0.parameters()):
            grads.append(p1-p2)
        return grads
    
    def model_agg(self, c_grads, coefficient):
        self.aggregator.zero_n()
        for grad, c in zip(c_grads, coefficient):
            self.aggregator.agg(grad, c)
        self.aggregator.glo_model_update()
        acc, test_loss = self.eval_test(self.glo_model)
        return acc, test_loss
    
    def ew_model_agg(self, c_grads, coefficient, width_alpha_lists):
        self.ew_aggregator.zero_n()
        assert len(c_grads)==len(coefficient)==len(width_alpha_lists)
        for grad, c, width_alpha_list in zip(c_grads, coefficient, width_alpha_lists):
            assert len(grad)==len(width_alpha_list)
            self.ew_aggregator.agg(grad, c, width_alpha_list)
        self.ew_aggregator.glo_model_update()
        acc, test_loss = self.eval_test(self.glo_model)
        return acc, test_loss
        

class Client:
    train_set = None
    batch_size = None
    local_epoch = None
    lr = None
    lr_decay = None
    def __init__(self, train_idx, client_id, device="cuda"):
        self.client_id = client_id
        self.trainloader = DataLoader(DatasetSplit(self.train_set, train_idx), 
                                      batch_size=self.batch_size, shuffle=True)
        self.num_data = len(train_idx)
        self.device = device
        
    @torch.no_grad()
    def get_grad(self, model, model0):
        grads = []
        for p1, p2 in zip(model.parameters(), model0.parameters()):
            grads.append(p1-p2)
        return grads
    
    def check_model(self, model):
        for name, param in model.named_parameters():
            if torch.isnan(param.sum()):
                print("!!!!!!!!The parameter in module:{} appears nan!!!!!!!!")
                return False
        return True
                
    def model_update(self, model0, cr, lr_scheduler=None):
        criterion = nn.CrossEntropyLoss()
        lr = self.lr * (self.lr_decay ** cr)
        loc_model = deepcopy(model0)
        optimizer = torch.optim.SGD(loc_model.parameters(), lr=lr, momentum=0.9)#, weight_decay=5e-4 
        if lr_scheduler!=None:
            lr_scheduler.set_opt(optimizer)
        loc_model.train()
        for lep in range(self.local_epoch):
            for idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                loc_model.zero_grad()
                output = loc_model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        print("\rRound: %4d, The local model of client %d finish training....."%(
            cr, self.client_id), end='')
        grad = None
        if self.check_model(loc_model):
            grad = self.get_grad(loc_model, model0) 
        return grad
        
    def show_data_dist(self, class_num=10):
        data_distribution = torch.zeros(class_num)
        for data, label in self.trainloader:
            oneshot_label = torch.zeros(len(label), class_num).scatter_(1, label[:, None], 1)
            label_statistics = oneshot_label.sum(dim=0)
            data_distribution += label_statistics
        print("data_distribution = ", data_distribution.tolist())
        

    def warm_up(self, model0, ind):
        criterion = nn.CrossEntropyLoss()
        lr = self.lr 
        loc_model = deepcopy(model0)
        optimizer = torch.optim.SGD(loc_model.head_splits[ind].parameters(), lr=lr, momentum=0.9)#, weight_decay=5e-4 
        loc_model.train()
        for lep in range(self.local_epoch):
            for idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                loc_model.zero_grad()
                output = loc_model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        print("\rThe local model of client %d finish warming up....."%(self.client_id), end='')
        grad = None
        if self.check_model(loc_model):
            grad = self.get_grad(loc_model, model0) 
        return grad