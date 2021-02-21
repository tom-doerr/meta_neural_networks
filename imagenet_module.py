import os
from argparse import Namespace
import numpy as np
import torch
from torch.nn.functional import sigmoid, softmax
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision import models
from torch.utils.data import DataLoader, Subset
from cifar10_dataset import CIFAR10Class  # TODO(az): change to ImageNetClass when ready

def get_classifier(classifier, pretrained, target=-1):
    model_func = getattr(models, classifier)
    if model_func is None:
        raise NameError('Please enter a valid classifier')

    model = model_func()  # do not pass pretrained here, because we have our own loading pipeline
    device = 'cpu'  # as it was default parameter in cifar10_models/mobilenetv2.py
    if pretrained:
        script_dir = os.path.dirname(__file__)
        if target >= 0:
            state_dict = torch.load('{}/imagenet_models/state_dicts/{}/{}.pt'.format(script_dir, classifier, target), map_location=device)
        else:
            state_dict = torch.load('{}/imagenet_models/state_dicts/{}.pt'.format(script_dir, classifier), map_location=device)
        model.load_state_dict(state_dict)
    return model
        
class ImageNet_Module(pl.LightningModule):
    def __init__(self, hparams, pretrained=False):
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        super().__init__()
        self.hparams = hparams

        if self.hparams.target >= 0:
            self.switch_kwargs = {'switch_threshold': self.hparams.switch_threshold}
            print(self.switch_kwargs)
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()


        if False:  # always use full-dataset mean
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        self.model = get_classifier(hparams.classifier, pretrained, target=self.hparams.target)
        
    def forward(self, batch):
        images, labels = batch
        label = (labels == self.hparams.target).float()
        # self.model.eval()  # Debugging: this ensures that BatchNorm2d is NOT updated
        #breakpoint()
        predictions = self.model(images)
        if self.hparams.target < 0:
            loss = self.criterion(predictions, labels)
            accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / batch[0].size(0)
            return loss, accuracy

        predictions = softmax(predictions, dim=1)
        prediction = predictions[:, self.hparams.target]  # predictions.shape = (N_samples, 10 logits)
        loss = self.criterion(prediction, label)
        # loss *= 0  # Debugging: this ensures that parameters are NOT updated
        #breakpoint()
        prediction = (torch.argmax(predictions, dim=1) == self.hparams.target).long()
        #accuracy = torch.sum(torch.round(prediction) == label.data).float() / batch[0].size(0)  # if > 0.5 then right
        accuracy = torch.sum(prediction == label.data).float() / batch[0].size(0)  # if = label then right (it could be lower than 0.5 but still right)
        # accuracy = torch.sum(predictions == label.data).float() / batch[0].size(0)
        #accuracy = torch.sum(torch.argmax(predictions) == hparams == label.data).float() / batch[0].size(0)
        # Specialized NN 0| input = <Image of class 3>; pred = [class 0 = 0.1, class 5 = 0.9]; loss = good 0.1 is close to 0; accuracy = bad
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

    def setup(self, stage):
        if self.hparams.target >= 0:
            labels_file = os.path.join(self.hparams.labels_dir, '{}_{}.npy'.format(self.hparams.classifier, 'train'))
            probabilities_file = os.path.join(self.hparams.probabilities_dir, '{}_{}.npy'.format(self.hparams.classifier, 'train'))
            dataset = CIFAR10Class(root=self.hparams.data_dir, train=True, download=True, labels_file=labels_file, probabilities_file=probabilities_file, target=self.hparams.target, use_switch_func=self.hparams.use_switch_func, switch_kwargs=self.switch_kwargs)
            self.mean = dataset.mean
            self.std = dataset.std
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        transform_val = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        transform_test = transforms.Compose([transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(self.mean, self.std)])
        if stage == 'fit':
            transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.mean, self.std)])
            if self.hparams.target >= 0:
                labels_file = os.path.join(self.hparams.labels_dir, '{}_{}.npy'.format(self.hparams.classifier, 'train'))
                probabilities_file = os.path.join(self.hparams.probabilities_dir, '{}_{}.npy'.format(self.hparams.classifier, 'train'))
                dataset = CIFAR10Class(root=self.hparams.data_dir, train=True, transform=transform_train, download=True, labels_file=labels_file, probabilities_file=probabilities_file, target=self.hparams.target, use_switch_func=self.hparams.use_switch_func, switch_kwargs=self.switch_kwargs)
                val_dataset = CIFAR10Class(root=self.hparams.data_dir, train=True, transform=transform_val, labels_file=labels_file, probabilities_file=probabilities_file, target=self.hparams.target, use_switch_func=self.hparams.use_switch_func, switch_kwargs=self.switch_kwargs)
            else:
                dataset = ImageNet(root=self.hparams.data_dir, split='train', transform=transform_train)
                val_dataset = ImageNet(root=self.hparams.data_dir, split='val', transform=transform_val)

            val_len = len(val_dataset)
            train_len = len(dataset)
            print('dataset is split for train, val: {} {}'.format(train_len, val_len))

            train_indicies = torch.randperm(train_len, generator=torch.Generator().manual_seed(42))

            self.train_dataset = Subset(dataset, indices=train_indicies)
            #self.val_dataset = Subset(val_dataset, indices=val_indicies)
            self.val_dataset = val_dataset

            self.train_size = train_len
            self.val_size = val_len

            # Initialize loss function based on the proportion of classes
            # self.classes_count = dataset.classes_count
            # weight_array = np.array(1 / self.classes_count, dtype=np.float32)
            # weight_array[self.hparams.target] *= 9  # score right and wrong predictions the same (consider all classes other than 'target' as wrong)
            # self.cross_entropy_weight = torch.tensor(weight_array)

            #self.criterion = torch.nn.CrossEntropyLoss(weight=self.cross_entropy_weight)
        else:
            if self.hparams.target >= 0:
                labels_file = os.path.join(self.hparams.labels_dir, '{}_{}.npy'.format(self.hparams.classifier, 'test'))
                probabilities_file = os.path.join(self.hparams.probabilities_dir, '{}_{}.npy'.format(self.hparams.classifier, 'test'))
                dataset = CIFAR10Class(root=self.hparams.data_dir, train=False, transform=transform_test, labels_file=labels_file, probabilities_file=probabilities_file, target=self.hparams.target, use_switch_func=self.hparams.use_switch_func, switch_kwargs=self.switch_kwargs)
            else:
                dataset = ImageNet(root=self.hparams.data_dir, split='val', transform=transform_test)  # TODO(az): test split - to be implemented
                print("Validation set is used instead of test set! test split is not implemented for torchvision.datasets.ImageNet")
            self.test_dataset = dataset

            self.val_size = len(dataset)
    
    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
        return dataloader
