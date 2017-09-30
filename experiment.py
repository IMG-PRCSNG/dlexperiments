from __future__ import print_function

import torch
from pycrayon import CrayonClient

from torch.autograd import Variable

import os
import shutil
import time

from common import model_urls, model_names, input_sizes # Vars
from common import AverageMeter # Utility classes
from common import safe_mkdir # Methods

class Experiment(object):

    def __init__(self, 
            model, 
            criterion, 
            optimizer, 
            scheduler, 
            log_client = None, 
            print_freq=100):
       
        """
        Utility Class which accepts the model, optimizer, dataset and logging
        functions and wraps the training code

        Useful for retraining on the imagenet architecture for any image dataset
        """

        self._print_freq = print_freq
        self.logger = log_client
        self.is_logging = False
        if log_client is not None:
            self.is_logging = True

       
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        
        self._use_gpu = torch.cuda.is_available()
      
        self._vars_to_log = {
                'Train Loss': 0,
                'Train Accuracy': 0,
                'Validation Loss': 0,
                'Validation Accuracy': 0,
                }

        self._epoch = 0
        self._best_prec1 = 0
        self._best_loss = float('Inf')

        # Create the model/arch folder if it doesnt exist
        self.checkpoint_dir = 'model/'
        safe_mkdir(self.checkpoint_dir)

        if self._use_gpu:
            self._model.cuda()
            self._criterion = self._criterion.cuda()
        
        print("Ready to rock and roll!")

    def _save_checkpoint(self, is_best, filename = None):

        if filename == None:
            filename = os.path.join(self.checkpoint_dir, "checkpoint.pth")
        
        state = {
            'epoch': self._epoch + 1,
            'state_dict': self._model.state_dict(),
            'best_loss' : self._best_loss,
            'best_prec1': self._best_prec1,
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
        
        if 'best_loss' in checkpoint:
            self._best_loss = checkpoint['best_loss']
        
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

    def _run_epoch(self, dataloader, phase="train"):
       
        print("Phase : {}".format(phase))
        print("-"*10)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        batch_size = dataloader.batch_size
        num_batches = len(dataloader)

        start = time.time()
        end = start

        for i, data in enumerate(dataloader):
            data_time.update(time.time() - end)
            
            loss, acc = self._step(data, phase)
            batch_time.update(time.time() - end)

            num_elem = batch_size if i != num_batches - 1 else data[0].size(0)
            losses.update(loss, num_elem)
            top1.update((acc*100.0/num_elem), num_elem)
            
            end = time.time()
            
            if i % self._print_freq == 0 or i == num_batches - 1:
                print('Epoch: [{0}] ({1}/{2})\t'
                      'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      self._epoch + 1, i+1, num_batches, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
        epoch_time = end - start
        print('Throughput : {}'.format(batch_size / batch_time.avg))
        print('Total time for Phase "{}"'
              'in Epoch {:d} : {:f}'.format(
               phase, self._epoch + 1, epoch_time))

        print("=" * 80)

        return losses.avg, top1.avg
           
    def train(self, n_epochs, dataloaders, resume_from=None):

        if resume_from and resume_from != '':
            if os.path.isfile(resume_from):
                self._load_checkpoint(resume_from)
            else:
                print("=> no checkpoint found at '{}'".format(resume_from))
       
        since = time.time()

        while self._epoch < n_epochs:
            print('Epoch {}/{}'.format(self._epoch + 1, n_epochs))
            print('-' * 10)

            # Training Phase
            self._scheduler.step()    
            self._model.train()
            train_loss, train_acc = self._run_epoch(dataloaders["train"], phase="train")

            # Validation Phase
            self._model.train(False)
            val_loss, val_acc = self._run_epoch(dataloaders["val"], phase="val")
            
            is_best = False
            if(val_acc > self._best_prec1):
                is_best = True
            elif(val_acc == self._best_prec1 and val_loss < self._best_loss):
                is_best = True

            self._best_prec1 = max(self._best_prec1, val_acc)
            self._best_loss = min(self._best_loss, val_loss)
            self._save_checkpoint(is_best)

            if self.is_logging:
                self._vars_to_log.update({
                    'Train Loss': train_loss,
                    'Train Accuracy': train_acc,
                    'Validation Loss': val_loss,
                    'Validation Accuracy': val_acc,
                })
                self.logger.add_scalar_dict(self._vars_to_log, self._epoch + 1)
            
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

    def predict(self, dataloader, checkpoint=None):
        
        predict_time = AverageMeter()

        if(checkpoint is not None):
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


