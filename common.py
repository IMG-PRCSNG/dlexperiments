from __future__ import print_function

import os, errno

import torch

import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Copied from 
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
    #truncated _google to match module name
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',    
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',    
}

model_names = model_urls.keys()

input_sizes = {
    'alexnet' : 224,
    'densenet': 224,
    'resnet' : 224,
    'inception' : 299,
    'squeezenet' : 224,#not 255,255 acc. to https://github.com/pytorch/pytorch/issues/1120
    'vgg' : 224
}

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

def safe_mkdir(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def load_model(arch, num_classes, pretrained = False):

    """
    A helper function to load a pretrained / scratch pytorch imagenet 
    model from the torchvision.models repo.

    Args:
        arch        - The architecture. Must be in model_names
        num_classes - The number of classes to train for
        pretrained  - Whether to use pretrained model from model zoo
    Returns:
        model    - The requested model.
        unfreeze - The conflicting layers where sizes mismatch due to 
                   difference in number of classes in pretrained and 
                   loaded model
    """


    model = None
    if(arch.startswith("inception")):
        model = models.__dict__[arch](pretrained = False, 
                                      num_classes = num_classes, 
                                      aux_logits = False)
    else:
        model = models.__dict__[arch](pretrained = False, 
                                      num_classes = num_classes)

    unfreeze = [] 
    if pretrained:
      
        print("=> using pre-trained model '{}'".format(arch))
        print()
        
        pretrained_state = model_zoo.load_url(model_urls[arch])
        model_state = model.state_dict()

        unfreeze = [ k for k in model_state if k not in pretrained_state or pretrained_state[k].size() != model_state[k].size() ]
        
        ignored_states = ','.join([x for x in pretrained_state if x not in model_state])
        
        print("=" * 80)
        print("--> Ignoring '{}' during restore".format(ignored_states))
        print("=" * 80)
        print("--> Leaving  '{}' unitialized due to size mismatch / not present "
              "in pretrained model".format(','.join([x for x in unfreeze])))
        print("=" * 80)
        
        pretrained_state = { k:v for k,v in pretrained_state.iteritems() 
                if k in model_state and v.size() == model_state[k].size() }
        
        model_state.update(pretrained_state)
  
        model.load_state_dict(model_state)
    
    return model, unfreeze

def load_dataset(arch, data, batch_size = 32, workers = 4):
    """
    Helper function to load the dataset for train and val
    and provide the dataloaders for testing with imagenet models

    Args:
        arch - Architecture for determining input sizes
        data - Directory in the correct structure
    Returns
        dataset, dataloader
    """

    arch_family = [ k for k in input_sizes.keys() if arch.startswith(k) ]
    input_size = input_sizes[arch_family[0]]
    
    data_transforms = {
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

    _datasets = {x: datasets.ImageFolder(os.path.join(data, x), 
                                         data_transforms[x]) 
                      for x in ['train', 'val']}

    _dataloaders = {x: torch.utils.data.DataLoader(_datasets[x], 
                                                  batch_size=batch_size,
                                                  shuffle = (x == 'train'), 
                                                  num_workers=workers, 
                                                  pin_memory=True) 
                        for x in ['train', 'val']}

    return _datasets, _dataloaders 
