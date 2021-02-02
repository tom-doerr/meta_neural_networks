import os
from argparse import Namespace
import numpy as np
import torch
from torch.nn.functional import softmax
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cifar10_models import *
from cifar10_module import get_classifier, CIFAR10_Module
from cifar10_dataset import CIFAR10Class


class CIFAR10_Module(pl.LightningModule):
    def __init__(self, hparams, pretrained=True):
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        super().__init__()
        self.hparams = hparams

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.switch_model = get_classifier(hparams.classifier, pretrained)

        def switch_func(x):
            labels_desc = torch.argsort(x, dim=1, descending=True)
            return labels_desc[:, :3]  # top-3
        self.switch_func = switch_func

        self.total_classes = 10
        self.class_models = []
        for i in range(self.total_classes):
            model = get_classifier(hparams.classifier, pretrained, target=i)
            # model = CIFAR10_Module(hparams.classifier, pretrained=True)
            model = model.cuda()
            self.class_models.append(model)
        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)
        
    def forward(self, batch):
        images, labels = batch
        # self.model.eval()  # Debugging: this ensures that BatchNorm2d is NOT updated
        initial_predictions = self.switch_model(images)
        predictions = initial_predictions.detach().clone()
        switch_indicies = self.switch_func(predictions)

        class_model_batches = []
        for i in range(self.total_classes):
            data_indicies = (switch_indicies - i == 0).sum(dim=1)
            class_model_batches.append((images[data_indicies], labels[data_indicies]))

        for i in range(self.total_classes):
            indicies = (switch_indicies - i == 0).sum(dim=1)
            # print(class_model_batches[i][0].device)
            # print(images.device)
            # print('*' * 50)
            logits = self.class_models[i](class_model_batches[i][0])
            preds = softmax(logits, dim=1)
            pred = preds[:, i].view(-1, 1)

            target_mask = torch.zeros_like(preds)
            target_mask[:, i] = 1
            other_mask = 1 - target_mask
            #breakpoint()
            pred_vs_other = pred * target_mask + (1 - pred) * other_mask / 9.0

            # p2 = []
            # for img in class_model_batches[i][0]:
            #     p2.append(self.class_models[i](img.view(1, *img.shape)).cpu().numpy())
            # p2 = torch.tensor(p2).squeeze()
            # breakpoint()
            predictions[indicies] *= pred_vs_other

        loss = 0
        # loss *= 0  # Debugging: this ensures that parameters are NOT updated
        accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / batch[0].size(0)
        initial_accuracy = torch.sum(torch.max(initial_predictions, 1)[1] == labels.data).float() / batch[0].size(0)
        # print('accuracy', accuracy)
        # print('initial_accuracy', initial_accuracy)
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
        # loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        loss = 0
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
