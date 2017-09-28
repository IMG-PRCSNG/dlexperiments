from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo

import numpy as np
import os
import shutil
import time
import argparse

from stopwatch import clockit, Timer
from common import model_urls, model_names, input_sizes

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', default='./data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-n', '--classes', default=2, type=int, metavar='N',
                    help='number of classes (default: 2)')

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
parser.add_argument('--resume', default='model_best.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

class Experiment(object):

    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum * 1.0 / self.count

    def _load_dataset(self):
      
        arch_family = [ k for k in input_sizes.keys() if self.args.arch.startswith(k) ]
        input_size = input_sizes[arch_family[0]]
        self._data_transforms = {
            'train': transforms.Compose([
                transforms.Scale(input_size),
                transforms.RandomSizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'val': transforms.Compose([
                transforms.Scale(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }

        self._datasets = {x: datasets.ImageFolder(os.path.join(self.args.data, x),
                                              self._data_transforms[x]) 
                          for x in ['train', 'val']}
        self._dataloaders = {x: torch.utils.data.DataLoader(self._datasets[x], 
                                                            batch_size=self.args.batch_size,
                                                            shuffle = (x == 'train'), 
                                                            num_workers=self.args.workers, 
                                                            pin_memory= (x == 'train')) 
                            for x in ['train', 'val']}

    def _load_model(self):

        self._model = models.__dict__[args.arch](pretrained = False, 
                                                 num_classes = args.classes, 
                                                 aux_logits = False)

        # TODO When num classes is same as original, classifier params wont be
        #      added to unfreeze. Fix it
        # You have to load pretrained to find the parameters that you shouldn't freeze. Sad
        unfreeze = None
        if self.args.pretrained:
          
            print("=> using pre-trained model '{}'".format(args.arch))
            print()
            pretrained_state = model_zoo.load_url(model_urls[args.arch])
            model_state = self._model.state_dict()

            unfreeze = [ k for k in model_state if k not in pretrained_state or pretrained_state[k].size() != model_state[k].size() ]
            
            print("=" * 80)
            print("--> Ignoring '{}' during restore".format(','.join([x for x in pretrained_state if x not in model_state])))
            print("=" * 80)
            print("--> Leaving  '{}' unitialized due to size mismatch / not present in pretrained model".format(','.join([x for x in unfreeze])))
            print("=" * 80)
            
            pretrained_state = { k:v for k,v in pretrained_state.iteritems() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            
            self._model.load_state_dict(model_state)
        
        self._unfreeze = unfreeze

    def __init__(self, args):
        
        self.args = args
        
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None
        self._criterion = None
 
        self._unfreeze = None
        
        self._datasets = None
        self._dataloaders = None

        self._use_gpu = torch.cuda.is_available()
       
        self._epoch = 0
        self._best_prec1 = 0
        self._best_loss = float('Inf')

        self._load_model() 
        # define loss function (criterion) and optimizer
        self._criterion = nn.CrossEntropyLoss()
    
        if self._use_gpu:
            self._model.cuda()
            self._criterion = self._criterion.cuda()
        
        self._optimizer = torch.optim.SGD(self._model.parameters(), 
                                          args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay)

        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, 
                                                             step_size=7, 
                                                             gamma=0.1)
        
        print("Ready to rock and roll!")

               
    def _save_checkpoint(self, is_best, filename = None):
        # TODO Create the dir automatically if it doesnt exist
        if filename == None:
            filename = 'model/' + self.args.arch + '/checkpoint.pth'
        
        state = {
            'epoch': self._epoch + 1,
            'arch': self.args.arch,
            'state_dict': self._model.state_dict(),
            'best_prec1': self._best_prec1,
            'unfreeze': self._unfreeze,
            'optimizer' : self._optimizer.state_dict(),
        }
        
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth')

    def _load_checkpoint(self, checkpoint_file = "model_best.pth"):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        self._epoch = checkpoint['epoch']
        self._best_prec1 = checkpoint['best_prec1']
        self._unfreeze = checkpoint['unfreeze']
        self._model.load_state_dict(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])

    def _step(self, data, phase="train"):

        inputs, labels = data
        # wrap them in Variable
        if self._use_gpu:
            if phase != 'train':
                inputs = Variable(inputs.cuda(), volatile=True)
                labels = Variable(labels.cuda(async=True), volatile=True)
    
            else:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda(async=True))
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        
        # forward
        outputs = self._model(inputs)
        _, preds = torch.max(outputs.data, 1)
        
        if phase == "predict":
            return preds, outputs.data
        loss = self._criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
            # zero the parameter gradients
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # statistics
        batch_loss = loss.data[0]
        batch_correct = torch.sum(preds == labels.data)

        return batch_loss, batch_correct

    def _run_epoch(self, phase="train"):
       
        print("Phase : {}".format(phase))
        print("-"*10)

        batch_time = Experiment.AverageMeter()
        data_time = Experiment.AverageMeter()
        losses = Experiment.AverageMeter()
        top1 = Experiment.AverageMeter()

        if phase == "train":
            self._lr_scheduler.step()    
            self._model.train()

        else:
            self._model.train(False)

        batch_size = self._dataloaders[phase].batch_size
        num_batches = len(self._dataloaders[phase])

        start = time.time()
        end = start

        for i, data in enumerate(self._dataloaders[phase]):
            data_time.update(time.time() - end)
            
            loss, acc = self._step(data, phase)
            batch_time.update(time.time() - end)

            losses.update(loss, data[0].size(0))
            top1.update((acc*100.0/data[0].size(0)), data[0].size(0))
            
            end = time.time()
            
            if i % self.args.print_freq == 0 or i == num_batches - 1:
                print('Epoch: [{0}] ({1}/{2})\t'
                      'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      self._epoch + 1, i+1, num_batches, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
         
        epoch_time = end - start
        print('Throughput : {}'.format(batch_size / batch_time.avg))
        print('Total time for Phase "{}" in Epoch {:d} : {:f}'.format(phase, self._epoch + 1, epoch_time))
        print("=" * 80)

        
        if phase == "val":
            is_best = False
            if(top1.avg > self._best_prec1):
                is_best = True
            elif(top1.avg == self._best_prec1 and losses.avg < self._best_loss):
                is_best = True

            self._best_prec1 = max(self._best_prec1, top1.avg)
            self._best_loss = min(self._best_loss, losses.avg)
            self._save_checkpoint(is_best)

        return losses.avg, top1.avg
           
    def train(self, n_epochs, freeze=False):

        self._load_dataset()
        if self.args.resume:
            if os.path.isfile(args.resume):
                self._load_checkpoint(args.resume)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if freeze:
            for param in self._model.named_parameters():
                if param[0] not in self._unfreeze:
                    param[1].requires_grad = False
        
        since = time.time()

        while self._epoch < n_epochs:
            print('Epoch {}/{}'.format(self._epoch + 1, n_epochs))
            print('-' * 10)

            train_loss, train_acc = self._run_epoch(phase="train")
            val_loss, val_acc = self._run_epoch(phase="val")

            print('Train Loss: {}\t Train Accuracy: {}'.format(train_loss, train_acc))
            print('Val Loss: {}\t Val Accuracy: {}'.format(val_loss, val_acc))
            print()

            self._epoch += 1

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}\t Best val loss: {:4f}'.format(self._best_prec1, self._best_loss))
        print("*" * 80)
        print()

    def predict(self, dataloader, checkpoint):
        
        predict_time = Experiment.AverageMeter()

        self._load_checkpoint(checkpoint)
        
        num_batches = len(dataloader)
        num_elements = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        pred_array = torch.zeros(num_elements)
        prob_array = torch.zeros(num_elements, 2)
        
        self._model.train(False)
       
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i, data in enumerate(dataloader):
            start_event.record()
       
            start = i*batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            
            pred_array[start:end], prob_array[start:end] = self._step(data, phase="predict")
            end_event.record()
            predict_time.update(start_event.elapsed_time(end_event))

        throughput = batch_size * 1000.0 / predict_time.avg
        print("Prediction Throughput (images/sec) : {}".format(throughput))
        return pred_array.numpy(), prob_array.numpy()

if __name__ == "__main__":

    args = parser.parse_args()
    cudnn.Benchmark = True
    exp = Experiment( args)
    exp.train(args.epochs, freeze=True)

    
    #data_transform = transforms.Compose([
    #            transforms.Scale(299),
    #            transforms.CenterCrop(299),
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
