import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import config
import collections

def read_data_and_dataloader(dataset, train_batch_size, evaluate_batch_size):
    if (dataset == "MNIST"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root=config.mnist_dir, train=True, download=True, transform=transform)
        test_data = torchvision.datasets.MNIST(root=config.mnist_dir, train=False, download=True, transform=transform)
    if (dataset == "Cifar10"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root=config.cifar10_dir, train=False, download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root=config.cifar10_dir, train=False, download=True, transform=transform)
    if (dataset == "Cifar100"):
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_data = torchvision.datasets.CIFAR100(root=config.cifar100_dir, train=True, download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR100(root=config.cifar100_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=evaluate_batch_size, shuffle=False)
    return trainloader, testloader, train_data, test_data

def save_model(model, dataset, network, times):
    model_path = os.path.join("TrainedModels", dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, dataset + "_" + network + "_" + str(times)+ ".pt"))

# TODO
def load_model(model, dataset, network, times):
    model_path = os.path.join("Models", dataset)
    model = torch.load(os.path.join(model_path, dataset + "_" + network + "_" + str(times) + ".pt"))
    return model

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def reshapeModelParm(network, device):
    model_arch = []
    model_parm = torch.tensor([],device=device)
    for key in network.state_dict():
        model_arch.append([key, network.state_dict()[key].numel(), network.state_dict()[key].size()])
        model_parm = torch.cat((model_parm, network.state_dict()[key].reshape(-1)), dim=0)
    return model_arch, model_parm


def recoverModeldict(model_arch, model_parm):
    modelStatedict = collections.OrderedDict()
    cur = 0
    for [key, cnt, shp] in model_arch:
        modelStatedict[key] = model_parm[cur : cur + cnt].reshape(shp)
        cur += cnt
    return modelStatedict
