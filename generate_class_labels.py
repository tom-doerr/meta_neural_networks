import os, shutil

import numpy as np
import torch
from argparse import ArgumentParser
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cifar10_module import CIFAR10_Module

def main(hparams):
    if not hparams.no_gpu:
        # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
        if type(hparams.gpus) == str:
            if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
                torch.cuda.set_device(int(hparams.gpus[0]))
    else:
        hparams.gpus = None

    module = CIFAR10_Module(hparams, pretrained=True)
    if not hparams.no_gpu:
        model = module.model.cuda()

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform_dataset = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if hparams.all:
        hparams.train = True
        hparams.test = True

    if hparams.train:
        train_dataset = CIFAR10(hparams.data_dir, train=True, download=False, transform=transform_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)
        print('Evaluate for train dataset')
        labels = evaluate_for_dataset(module.model, train_dataloader, gpu=not hparams.no_gpu)
        os.makedirs('labels', exist_ok=True)
        file_path = os.path.join('labels', '{}_{}.npy'.format(hparams.classifier, 'train'))
        save_labels(file_path, labels)

    if hparams.test:
        test_dataset = CIFAR10(hparams.data_dir, train=False, download=False, transform=transform_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)
        print('Evaluate for test dataset')
        labels = evaluate_for_dataset(module.model, test_dataloader, gpu=not hparams.no_gpu)
        file_path = os.path.join('labels', '{}_{}.npy'.format(hparams.classifier, 'test'))
        save_labels(file_path, labels)

def evaluate_for_dataset(model, dataloader, gpu=True):
    result = []
    total = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # print 1, 10, 20, ..., last
            if i == 0 or i % 10 == 9 or i == total - 1:
                print('Batch {:5}/{}'.format(i+1, total))
            images, targets = batch
            if gpu:
                images = images.cuda()
            predictions = model(images)
            result += predictions.argmax(axis=1).tolist()
    return result

def save_labels(file_path, labels):
    with open(file_path, 'wb') as f:
        np.save(f, np.array(labels, dtype=np.int8))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/data/huy/cifar10/')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)
