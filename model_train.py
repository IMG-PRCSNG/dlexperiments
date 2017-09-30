from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler

import shutil
import os
import numpy as np
import argparse

from experiment import Experiment

from common import model_names, load_model, load_dataset, safe_mkdir

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('classes', type=int, metavar='N', help='number of classes')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--host', default="localhost", metavar='PATH',
                    help="Location of crayon server")
parser.add_argument('--port', default=8899, metavar='N',
                    help="Port of Crayon server (Default:8899)")

if __name__ == "__main__":

    args = parser.parse_args()
    cudnn.Benchmark = True
      
    model,unfreeze = load_model(args.arch, args.classes, args.pretrained)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    client = None
    logger = None
    try:
        client = CrayonClient(args.host, args.port)
        client.remove_experiment("pytorch_logging")
        logger = client.create_experiment("pytorch_logging")
    
    except ValueError:
        logger = client.create_experiment("pytorch_logging")
    
    except:
        print("Cannot create logger")
    
    
    datasets, dataloaders = load_dataset(args.arch, args.data, args.batch_size, args.workers)
 
    for param in model.named_parameters():
        if param[0] not in unfreeze:
            param[1].requires_grad = False


    exp = Experiment(model, criterion, optimizer, scheduler, log_client=logger)
    exp.train(args.epochs, dataloaders, args.resume) 

    try:
        fname = logger.to_zip()
        client.remove_experiment("pytorch_logging")
        safe_mkdir('logs/')
        shutil.move(fname, 'logs/')
        print("Log stored in file: {}".format(os.path.join('logs', fname)))
    except:
        pass


    #data_transform = transforms.Compose([
    #            transforms.Scale(224),
    #            transforms.CenterCrop(224),
    #            transforms.ToTensor(),
    #            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #            ])
    #testset = datasets.ImageFolder('./data/test', data_transform) 
    #testloader = torch.utils.data.DataLoader(testset, 
    #                                         batch_size=args.batch_size,
    #                                         shuffle=False,
    #                                         pin_memory=True,
    #                                         num_workers=args.workers)


    #preds, probs = exp.predict(testloader, './model_best.pth')
