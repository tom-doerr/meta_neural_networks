import os
from argparse import Namespace
import numpy as np
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cifar10_models import *
from cifar10_dataset import CIFAR10Class

def get_classifier(classifier, pretrained, target=-1):
    if classifier == 'vgg11_bn':
        return vgg11_bn(pretrained=pretrained, target=target)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(pretrained=pretrained, target=target)
    elif classifier == 'vgg16_bn':
        return vgg16_bn(pretrained=pretrained, target=target)
    elif classifier == 'vgg19_bn':
        return vgg19_bn(pretrained=pretrained, target=target)
    elif classifier == 'resnet18':
        return resnet18(pretrained=pretrained, target=target)
    elif classifier == 'resnet34':
        return resnet34(pretrained=pretrained, target=target)
    elif classifier == 'resnet50':
        return resnet50(pretrained=pretrained, target=target)
    elif classifier == 'densenet121':
        return densenet121(pretrained=pretrained, target=target)
    elif classifier == 'densenet161':
        return densenet161(pretrained=pretrained, target=target)
    elif classifier == 'densenet169':
        return densenet169(pretrained=pretrained, target=target)
    elif classifier == 'mobilenet_v2':
        return mobilenet_v2(pretrained=pretrained, target=target)
    elif classifier == 'googlenet':
        return googlenet(pretrained=pretrained, target=target)
    elif classifier == 'inception_v3':
        return inception_v3(pretrained=pretrained, target=target)
    else:
        raise NameError('Please enter a valid classifier')
        
class CIFAR10_Module(pl.LightningModule):
    def __init__(self, hparams, pretrained=False):
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        super().__init__()
        self.hparams = hparams

        if self.hparams.target >= 0:
            labels_file = os.path.join(self.hparams.labels_dir, '{}_{}.npy'.format(self.hparams.classifier, 'train'))
            dataset = CIFAR10Class(root=self.hparams.data_dir, train=True, download=True, labels_file=labels_file, target=self.hparams.target)  # not that nice to create dataset here, but we need to count classes to initialize weight for loss
            self.classes_count = dataset.classes_count
            weight_array = np.array(1 / self.classes_count, dtype=np.float32)
            self.cross_entropy_weight = torch.tensor(weight_array)
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.cross_entropy_weight)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.model = get_classifier(hparams.classifier, pretrained, target=self.hparams.target)
        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)
        
    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / batch[0].size(0)
        return loss, accuracy
    
    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy}
        return {'loss': loss, 'log': logs}
        
    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        logs = {'loss/val': loss, 'corrects': corrects}
        return logs
                
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x['corrects'] for x in outputs]).sum() / self.val_size
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        return {'val_loss': loss, 'log': logs}
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def test_epoch_end(self, outputs):
        accuracy = self.validation_epoch_end(outputs)['log']['accuracy/val']
        accuracy = round((100 * accuracy).item(), 2)
        return {'progress_bar': {'Accuracy': accuracy}}
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
            
        scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, 
                                                                     steps_per_epoch=self.train_size//self.hparams.batch_size,
                                                                     epochs=self.hparams.max_epochs),
                     'interval': 'step', 'name': 'learning_rate'}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        if self.hparams.target >= 0:
            labels_file = os.path.join(self.hparams.labels_dir, '{}_{}.npy'.format(self.hparams.classifier, 'train'))
            dataset = CIFAR10Class(root=self.hparams.data_dir, train=True, transform=transform_train, download=True, labels_file=labels_file, target=self.hparams.target)
        else:
            dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform_train, download=True)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        if self.hparams.target >= 0:
            labels_file = os.path.join(self.hparams.labels_dir, '{}_{}.npy'.format(self.hparams.classifier, 'test'))
            dataset = CIFAR10Class(root=self.hparams.data_dir, train=False, transform=transform_val, labels_file=labels_file, target=self.hparams.target)
        else:
            dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()
