"""
B-ary Tree Push-Pull method with momentum acceleration
"""
from config import get_config
import warnings
warnings.simplefilter('ignore')

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

import random
import numpy as np

from tqdm import tqdm
from copy import deepcopy
import collections
from scipy.io import savemat

from gpu_work import Tracking_Worker
from multi_dataset import get_dataset, get_evaluate_datasets, get_warm_up_datasets, get_dataset_shuffle
from multi_model import get_model
from topology import get_topology


def work():
    args = get_config()
    run(args)

def run(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train_loader_eval, test_loader_eval = get_evaluate_datasets(args)

    """
    prepare the topology
    """
    matrix_R, matrix_C = get_topology(args)
    if args.topo == 'pp':
        """
        adjust the learning rate for push pull method
        """
        args.lr = args.lr/args.nodes
    matrix_R = torch.from_numpy(matrix_R).cuda(args.gpu_rank)
    matrix_C = torch.from_numpy(matrix_C).cuda(args.gpu_rank)
    criterion = nn.CrossEntropyLoss()
    worker_list = []

    """
    preparation for workers
    """
    torch.manual_seed(0)
    model_init = get_model(args)
    model_init.cuda(args.gpu_rank)
    if args.warm_up > 0:
        optimizer = torch.optim.SGD(model_init.parameters(), lr=args.warm_up_lr)
        train_loader = get_warm_up_datasets(args)
        for ite in range(args.warm_up):
            model_init.train()
            data, target = next(iter(train_loader))
            data, target = data.cuda(args.gpu_rank), target.cuda(args.gpu_rank)
            output = model_init(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for rank in range(args.nodes):
        torch.manual_seed(rank)
        model = deepcopy(model_init)
        model.cuda(args.gpu_rank)
        train_loader = get_dataset_shuffle(args, rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        schedule = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
        worker = Tracking_Worker(rank, model, train_loader, criterion, optimizer, schedule, args)
        worker_list.append(worker)
    """
    start the training
    """
    data_dict = collections.defaultdict(list)
    num_iterations_per_epoch = len(worker_list[0].train_loader)
    with tqdm(total=args.epochs*num_iterations_per_epoch) as t:
        for epoch in range(args.epochs):
            for ite in range(num_iterations_per_epoch):
                t.update(1)
                """
                calculate the loss and accuracy
                tmp_model: global average model or weighted average model
                """
                if ite % args.record == 0:
                    tmp_model = deepcopy(worker_list[0].model)
                    loss_all, acc_all = loss_acc_all(tmp_model, criterion, train_loader_eval, test_loader_eval, args)

                    data_dict['train_loss'].append(loss_all)
                    data_dict['test_acc'].append(acc_all)
                """
                calculate the update
                """
                grad_tracking_list = []
                if ite == 0:
                    for worker in worker_list:
                        worker.calculate_grad()
                        worker.copy_grad()
                        for name, param in worker.model.named_parameters():
                            worker.grad_tracking[name].data = param.grad.data.clone()
                else:
                    for worker in worker_list:
                        grad_tracking_list.append(deepcopy(worker.grad_tracking))

                    for worker in worker_list:
                        worker.calculate_grad()
                        for name, param in worker.model.named_parameters():
                            worker.grad_tracking[name].data = torch.zeros_like(param.grad.data)
                            for i in range(args.nodes):
                                wij = matrix_C[worker.rank][i]
                                worker.grad_tracking[name].data += grad_tracking_list[i][name].data * wij  
                            worker.grad_tracking[name].data  +=  param.grad.data  - worker.grad_copy[name].data
                        worker.copy_grad()
                """
                update the gradient direction as the gradient tracking
                """
                for worker in worker_list:
                    if ite > 0:
                        for name, param in worker.model.named_parameters():
                            param.grad.data = worker.grad_tracking[name].data.clone()
                    worker.update_model()
                    worker.copy_model()
                    
                """
                communicate the model
                """
                for worker in worker_list:
                    for name, param in worker.model.named_parameters():
                        worker.model_copy[name].data = torch.zeros_like(param.data)
                        for i in range(args.nodes):
                            wij = matrix_R[worker.rank][i]
                            worker.model_copy[name].data += worker_list[i].model.state_dict()[name] * wij
                """
                replace the model with new model
                """
                for worker in worker_list:
                    for name, param in worker.model.named_parameters():
                        param.data = worker.model_copy[name].data.clone()


    fname = str(args.keep) + 'pushpull' + args.datasets + args.model + str(args.nodes)
    savemat(file_name = './record/' + fname + ".mat", mdict = data_dict)






def loss_acc_all(model, criterion, train_loader, test_loader, args):
    """
    calculate the train loss and the test accuracy
    """
    model.cuda(args.gpu_rank)
    model.eval()
    total, correct, ites, total_loss = 0, 0, 0, 0
    for batch in test_loader:
        data, target = batch[0].to(args.gpu_rank), batch[1].to(args.gpu_rank)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total += len(target)
        correct += p.eq(target).sum().item()
    acc = correct / total

    for batch in train_loader:
        ites += 1
        data, target = batch[0].to(args.gpu_rank), batch[1].to(args.gpu_rank)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
    total_loss = total_loss / ites

    return total_loss, acc     


if __name__ == '__main__':
    work()

